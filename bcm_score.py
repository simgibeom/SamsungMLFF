
import os
import sys
import glob
import time
import numpy as np
from ase import Atoms, Atom
from ase.io import read, write, Trajectory
from theforce.util.parallel import mpi_init 
from nonactive_rbcm_4score import *

process_group = mpi_init()
rank = 0

#diff = '0$diff' # 0.4 => 04 | 0.01 => 001
diff = '0005'

start_dir = '.'
file_pattern = f'model-*-{diff}.pckl'

file_paths = []
for root, dirs, files in os.walk(start_dir):
    for file in glob.glob(os.path.join(root, file_pattern)):
        #print("Found:",file)
        file_paths += [f'{file}']
#print(f'\n === file_paths ===\n {file_paths}')
kernel_model_dict = {}
for i in range(len(file_paths)):
    kernel_model_dict[f'key{i+1}'] = file_paths[i]
if rank == 0:
    print(f'\n === kernel_model_dict ===\n {kernel_model_dict}')

bcm_calc = BCMCalculator(process_group=process_group,
                         kernel_model_dict=kernel_model_dict)
if distrib.is_initialized():
   rank = distrib.get_rank ()

atoms = read('v_Si_27.traj', index=slice(None))

t1 = time.time()
error_list = []
f_error_list = []
num_list = []
for iat, exact in enumerate(atoms):
    atom = exact.copy()
    atom.calc = bcm_calc 
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
