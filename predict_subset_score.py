import torch
from ase.calculators.socketio import SocketClient
from ase.io import read
from theforce.calculator.socketcalc import SocketCalculator
from theforce.calculator.active import ActiveCalculator, FilterDeltas
from theforce.util.parallel import mpi_init
from ase.build import bulk
from ase.md.npt import NPT
from ase import units
from ase.calculators.emt import EMT
from ase.io import read, Trajectory

from theforce.regression.gppotential import PosteriorPotential, GaussianProcessPotential
from theforce.similarity.similarity import SimilarityKernel
from theforce.similarity.sesoap import *

import numpy as np

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

process_group = mpi_init()
common = dict(ediff=0.1, fdiff=0.1, process_group=process_group)
kernel_kw = dict(lmax=3, nmax=3, exponent=4, cutoff=6.0)
test = Trajectory('v_Si.traj')

calc = ActiveCalculator(covariance='model.pckl',
                        calculator=None,
                        logfile='active.log4',
                        pckl='model2.pckl/',
                        tape='model.sgpr',
                        kernel_kw=kernel_kw,
                        **common)
error_list = []
f_error_list = []
num_list = []
for i in range(len(test)):
	atom = test[i].copy()
	atom.calc = calc
	error = rmse_square(predict=atom, exact=test[i])
	f_error = axis_wise_rmse_of_force_square(predict=atom, exact=test[i])
	error_list += [error]
	f_error_list += [f_error]
	num_list += [len(atom)]
	print(f'\n ########### TEST_{i:04} ############ ')
	print(f' {i}-th config energy           : {atom.get_potential_energy()}')
	print(f' {i}-th config RMSE**2          : {error}')
	print(f' {i}-th config RMSE**2 of force : {f_error}')
print(f'\n===> RMSE of per-atom energy : {np.sqrt(np.average(np.array(error_list)))}')
print(f'===> Axis-wise RMSE of force : {np.sqrt(sum(f_error_list)/(3*sum(num_list)))}\n')
