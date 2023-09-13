diff=1 # ex) 1 ==> 0.1 | 01 ==> 0.01 
old_diff=4
touch diff-0$diff.log
for i in {0..29} ; do
cd task_subset-$i/
mkdir re-0$diff
cp subset-$i.traj v_Si.traj train.py predict_subset_score.py re-0$diff/
echo "  "
echo " ##### Training the model-$i using the subset-$i ... ##### "
echo "  "
cd re-0$diff/

cat >train.py <<!
from theforce.calculator.active import ActiveCalculator, FilterDeltas
from theforce.util.parallel import mpi_init
from ase import Atoms
from ase.io import read, write, Trajectory

common = dict(ediff=0.$diff, fdiff=0.$diff, ediff_tot=0.$diff, process_group=mpi_init())
kernel_kw = dict(lmax=3, nmax=3, exponent=4, cutoff=4.0)

ML_calc = ActiveCalculator(covariance='../model-$i-0$old_diff.pckl',
                           calculator=None,
                           logfile='active-$i-0$diff.log',
                           pckl='model-$i-0$diff.pckl/',
                           tape='model-$i-0$diff.sgpr',
                           kernel_kw=kernel_kw,
                           **common)
data = Trajectory('subset-$i.traj')

ML_calc.include_data(data)
!
mpirun -np 4 python train.py

cat >predict_subset_score.py <<!
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


process_group = mpi_init()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

test = Trajectory('v_Si.traj')
common = dict(ediff=0.1, fdiff=0.1, process_group=process_group)
kernel_kw = dict(lmax=3, nmax=3, exponent=4, cutoff=4.0)

calc = ActiveCalculator(covariance='model-$i-0$diff.pckl',
                        calculator=None,
                        logfile='active-$i-0$diff-p.log',
                        pckl='model-$i-0$diff-p.pckl/',
                        tape='model-$i-0$diff-p.sgpr',
                        kernel_kw=kernel_kw,
                        **common)
t1 = time.time()
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
    if rank == 0:
        msg = [f'\n ########### TEST_{i:04} ############ ']
        msg += [f' {i}-th config energy           : {atom.get_potential_energy()}']
        msg += [f' {i}-th config RMSE**2          : {error}']
        msg += [f' {i}-th config RMSE**2 of force : {f_error}']
        message(msg)

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
!
mpirun -np 2 python predict_subset_score.py
echo "### ediff / fdiff / ediff_tot = 0.$diff ###" >> ../../diff-0$diff.log
echo "  "  >> ../../diff-0$diff.log
score=`grep SCORE predict.log`

echo " "
echo "   [Model-$i] $score "  
echo " "

echo "[Model-$i] $score " >> ../../diff-0$diff.log
cd ../../
done
