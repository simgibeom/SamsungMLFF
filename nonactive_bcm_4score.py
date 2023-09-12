import os
import time
import ase 

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes

import theforce.distributed as distrib
from theforce.descriptor.atoms import TorchAtoms, Distributer

from theforce.util.util import date
from theforce.regression.gppotential import PosteriorPotentialFromFolder

class BCMCalculator(Calculator):
    """
        Bayesian Committee Machine (BCM)
        E(*|D) = \sum_i [E(*|D_i) / cov(*|D_i])]  x [\sum_i 1/cov(*|D_i)]^{-1}
    """
    implemented_properties =  ["energy", "forces", "stress"]

    def __init__(
        self,
        process_group=None,
        kernel_model_dict=None,
        logfile='bcm_nonactive.log',
		pckl='bcm1.pckl',
		tape='bcm1.sgpr',
        **kwargs,
    ):
        Calculator.__init__ (self)
        self.process_group = process_group
        self.logfile = logfile
        self.rank = 0 
        world_size = 1
        if distrib.is_initialized():
            world_size = distrib.get_world_size()
            self.rank = distrib.get_rank ()
        self.distrib = Distributer(world_size)
        self.K_sm = {}
        self.model_dict = {}
        self.step = 0

        self.cutoff = 6.0
        self.key_list = []
        if kernel_model_dict is not None:
            for key in kernel_model_dict:
                self.model_dict[key] = \
                    PosteriorPotentialFromFolder(
                    kernel_model_dict[key],
                    load_data=True,
                    update_data=False,
                    group=self.process_group,
                    distrib=self.distrib
                    )
                self.key_list.append(key)
    
    def calculate (self, _atoms=None, properties=["energy"], system_changes=all_changes):
        timings = [time.time()]

        if type(_atoms) == ase.atoms.Atoms:
            atoms = TorchAtoms(ase_atoms=_atoms, ranks=self.distrib)
            uargs = {
                "cutoff": self.model_dict[self.key_list[-1]].cutoff,
                "descriptors": self.model_dict[self.key_list[-1]].gp.kern.kernels,
            }
            self.to_ase = True
        else:
            atoms = _atoms
            uargs = {}
            self.to_ase = False

        if _atoms is not None and self.process_group is not None:
            atoms.attach_process_group (self.process_group)
        
        Calculator.calculate (self, atoms, properties, system_changes)
        self.atoms.update (
            posgrad=True, cellgrad=True, forced=True, dont_save_grads=True, **uargs
        )
        timings.append (time.time())

        # kernel: covariance values between configuration(atoms) and inducing (LCE)
        for key in self.model_dict:
            self.K_sm[key] = self.model_dict[key].gp.kern (self.atoms,
                                                           self.model_dict[key].X)
        
        energy, covloss_max = self.update_results ()
        timings.append (time.time())

        # Non-active learning
        self.covlog = f"{covloss_max}"
        self.post_calculate (timings)
        

    def update_results(self, retain_graph=False):
        #natoms = self.atoms.positions.shape[0]
        energy = 0.0
        mean = 0.0
        covloss_inv = 0.0
        covloss_max = float('inf')
        #nmodel = len(self.K_sm)

        ''' ver1
        for key in self.K_sm:
            covloss = self.get_covloss (key, self.K_sm[key])
            covloss_inv += 1.0/covloss.max()
            enr_key = self.K_sm[key] @ self.model_dict[key].mu 
            mean_key = self.model_dict[key].mean(self.atoms)
            energy += (enr_key.sum())/covloss.max()
            mean   += (mean_key)/covloss.max()
            covloss_max = min (covloss_max, covloss.max())
		''' 

        for key in self.K_sm:
            covloss = self.get_covloss (key, self.K_sm[key])
            covmax = covloss.max()
            covmax2 = covmax*covmax
            covloss_inv += 1.0/covmax2
            enr_key = self.K_sm[key] @ self.model_dict[key].mu 
            mean_key = self.model_dict[key].mean(self.atoms)
            energy += (enr_key.sum())/covmax2
            mean   += (mean_key)/covmax2
            covloss_max = min (covloss_max, covmax)

        energy = energy/covloss_inv 
        mean   = mean/covloss_inv 
        if self.atoms.is_distributed:
            distrib.all_reduce (energy)
        energy += mean     
        
        forces, stress = self.grads (energy, retain_graph=retain_graph)
        
        self.results["energy"] = energy.detach().numpy()
        self.results["forces"] = forces.detach().numpy()
        self.results["stress"] = stress.flat[[0,4,8,5,2,1]]
        maximum_force = abs (self.results["forces"]).max()
        self.maximum_force = f"{maximum_force}"

        return self.results["energy"], covloss_max
    

    def grads(self, energy, retain_graph=False):
        if energy.grad_fn:
            forces = -torch.autograd.grad(
                energy, self.atoms.xyz, retain_graph=True, allow_unused=True
            )[0]
            (cellgrad,) = torch.autograd.grad(
                energy, self.atoms.lll, retain_graph=retain_graph, allow_unused=True
            )
            if cellgrad is None:
                cellgrad = torch.zeros_like(self.atoms.lll)
        else:
            forces = torch.zeros_like(self.atoms.xyz)
            cellgrad = torch.zeros_like(self.atoms.lll)
        if self.atoms.is_distributed:
            distrib.all_reduce(forces)
            distrib.all_reduce(cellgrad)
        # stress
        stress1 = -(forces[:, None] * self.atoms.xyz[..., None]).sum(dim=0)
        stress2 = (cellgrad[:, None] * self.atoms.lll[..., None]).sum(dim=0)
        try:
            volume = self.atoms.get_volume()
        except ValueError:
            volume = -2  # here stress2=0, thus trace(stress) = virial (?)
        stress = (stress1 + stress2).detach().numpy() / volume
        return forces, stress

    
    def post_calculate(self, timings):
        energy = self.results["energy"]
        self.log (
            "{} {} {} {}".format (
            energy, self.atoms.get_temperature(), self.covlog, self.maximum_force
            )
        )
        self.step += 1
        
        if False: 
            delta_time = np.diff (timings)
            self.log (
                ("timings:" + (len(timings)-1)*"{:0.2g}").format (*delta_time)
                + f" total: {sum(delta_time):0.2g}"
            )


    def gather(self, x):
        if self.atoms.is_distributed:
            size = [s for s in x.size()]
            size[0] = self.atoms.natoms
            _x = torch.zeros(*size)
            _x[self.atoms.indices] = x
            distrib.all_reduce(_x)
            return _x
        else:
            return x


    def get_covloss(self, model_key, K_sm):
        b = self.model_dict[model_key].choli @ K_sm.detach().t()
        c = (b * b).sum(dim=0)
        
        if c.size(0) > 0:
            beta = (1 - c).clamp(min=0.0).sqrt()
        else:
            beta = c
        beta = self.gather(beta)
        vscale = []
        for z in self.atoms.numbers:
            if z in self.model_dict[model_key]._vscale:
                vscale.append(self.model_dict[model_key]._vscale[z])
            else:
                vscale.append(float("inf"))
        vscale = torch.tensor(vscale).sqrt()
        return beta * vscale
    


    def log(self, mssge, mode="a"):
        if self.logfile and self.rank == 0:
            with open(self.logfile, mode) as f:
                f.write(" {} {} {}\n".format( date(), self.step, mssge))
            # cov log
            if mode == "w" and False:
                with open("cov.log", mode) as f:
                    f.write("# covariance data\n")

###
import torch
import numpy as np
import time

from mpi4py import MPI
from ase.io import read, Trajectory
from theforce.calculator.active import ActiveCalculator, FilterDeltas
from theforce.util.parallel import mpi_init

def rmse_square(predict, exact):
    """
    This function only return about numerator if each RMSE of per-atom energy.
    predict and exact : Atoms object
    """
    return ((predict.get_potential_energy()-exact.get_potential_energy())/len(predict))**2

def axis_wise_rmse_of_force_square(predict, exact):
    """
    This function only return about numerator of each Axis-wise RMSE if force.
    predict and exact : Atoms object
    """
    f_p = predict.get_forces().copy()
    f_e = exact.get_forces().copy()
    return sum([( ((f_p[i][0]-f_e[i][0])**2) + ((f_p[i][1]-f_e[i][1])**2) + ((f_p[i][2]-f_e[i][2])**2) ) for i in range(len(f_p))])

def message(msg):
    with open("predict.log", "a") as file:
        for m in msg:
            file.write(m + '\n')
###

if __name__ == "__main__":
    import sys
    import glob
    from ase.io import read
    from theforce.util.parallel import mpi_init 

    kernel_model_dict = {'key1':'../bcm_sub-16/bcm_sub-16-01/model-16-01.pckl/',
                         'key2':'../bcm_sub-27/bcm_sub-27-01/model-27-01.pckl/', 
                         'key3':'../bcm_sub-32/bcm_sub-32-01/model-32-01.pckl/',
                         'key4':'../bcm_sub-64/bcm_sub-64-01/model-64-01.pckl/',
                         'key5':'../bcm_sub-214/bcm_sub-214-01/model-214-01.pckl/',
                         'key5':'../bcm_sub-510/bcm_sub-510-01/model-510-01.pckl/'}

    process_group = mpi_init()

    bcm_calc = BCMCalculator(
        process_group=process_group,
        kernel_model_dict=kernel_model_dict
    )

    rank = 0
    if distrib.is_initialized():
       rank = distrib.get_rank ()

    atoms = read('v_Si.traj', index=slice(None))
    
    t1 = time.time()
    error_list = []
    f_error_list = []
    num_list = []
    for iat, exact in enumerate(atoms):
        energy_FP = exact.get_potential_energy()
        forces_FP = exact.get_forces ()
        atom = exact.copy()
        atom.calc = bcm_calc 
        enr_ML = atom.get_potential_energy()
        frc_ML = atom.get_forces()
        natoms = len(exact)

        ###
        error = rmse_square(predict=atom, exact=exact)
        f_error = axis_wise_rmse_of_force_square(predict=atom, exact=exact)
        error_list += [error]
        f_error_list += [f_error]
        num_list += [len(atom)]
        if rank == 0:
            msg = [f'\n ########### TEST_{iat:04} ############ ']
            msg += [f' {iat}-th config energy           : {atom.get_potential_energy()}']
            msg += [f' {iat}-th config RMSE**2          : {error}']
            msg += [f' {iat}-th config RMSE**2 of force : {f_error}']
            message(msg)
            
            ediff = enr_ML - energy_FP
            fdiff = abs(frc_ML - forces_FP).mean()
            print('iconf', iat, natoms, ediff/natoms, fdiff, flush=True)
			#print ('iconf', iat, ediff, fdiff)
			#print ('enr', enr_ML-energy_FP, enr_ML, energy_FP)
			#print ('frc', frc_ML - forces_FP)    
    
    RMSE = np.sqrt(np.average(np.array(error_list)))
    Axis_wise_RMSE = np.sqrt(sum(f_error_list)/(3*sum(num_list)))
    lambda_force = 1/25
    t2 = time.time()
    
    if rank == 0:
        msg = [f'\n> RMSE of per-atom energy : {RMSE}']
        msg += [f'> Axis-wise RMSE of force : {Axis_wise_RMSE}\n']
        msg += [f'===> SCORE : {(RMSE + (lambda_force * Axis_wise_RMSE))*1000}\n']
        msg += [f'Time spent : {t2-t1}']
        message(msg)
