## STEP 1) diff = 0.4

diff=4 # ex) 1 ==> 0.1 | 01 ==> 0.01 
old_diff=None

for i in {182..231} ; do
mkdir task_subset-$i
cp 4_GB/subset-$i.traj train.py task_subset-$i/
echo "  "
echo "  ##### Training the model-$i using the subset-$i ... ##### "
echo "  ***  ediff / fdiff / ediff_tot = 0.$diff  *** " 
echo "  "
cd task_subset-$i/

cat >train.py <<!
from theforce.calculator.active import ActiveCalculator, FilterDeltas
from theforce.util.parallel import mpi_init
from ase import Atoms
from ase.io import read, write, Trajectory
import glob
import os

common = dict(ediff=0.$diff, fdiff=0.$diff, ediff_tot=0.$diff, process_group=mpi_init())
kernel_kw = dict(lmax=3, nmax=3, exponent=4, cutoff=4.0)
if type($old_diff) == str:
    if glob.glob(f"../model-$i-0$old_diff.pckl") == []:
        covariance = None
    else:
        covariance = '../model-$i-0$old_diff.pckl'
else:
    covariance = None
ML_calc = ActiveCalculator(covariance=covariance,
                           calculator=None,
                           logfile='active-$i-0$diff.log',
                           pckl='model-$i-0$diff.pckl/',
                           tape='model-$i-0$diff.sgpr',
                           kernel_kw=kernel_kw,
                           **common)
data = Trajectory('subset-$i.traj')

ML_calc.include_data(data)
!
mpirun -np 8 python train.py

cd ../
done

echo "  ##### Predicting using validation set through BCM ... ##### "

cat >bcm_score.py <<!
import os
import sys
import glob
import time
import numpy as np
from ase import Atoms, Atom
from ase.io import read, write, Trajectory
from theforce.util.parallel import mpi_init 
from nonactive_bcm_4score import *

diff = '0$diff' # 0.4 => 04 | 0.01 => 001
file_list = glob.glob("4_GB/subset-*.traj")
total_files_num = len(file_list)
kernel_model_dict = {}
for i in range(total_files_num):
    search_directory = os.getcwd()
    path = glob.glob(os.path.join(search_directory, "**", f'model-{i+182}-{diff}.pckl'), recursive=True)   
    print(path)
    if path != []:
        kernel_model_dict[f'key{i+1}'] = path[0]
process_group = mpi_init()
bcm_calc = BCMCalculator(process_group=process_group,
                         kernel_model_dict=kernel_model_dict)
rank = 0
if distrib.is_initialized():
   rank = distrib.get_rank ()

atoms = read('v_Si_N.traj', index=slice(None))

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
!
mpirun -np 8 python bcm_score.py
mkdir bcm-0$diff
cp bcm_score.py bcm-0$diff/ 
mv predict.log bcm-0$diff/


## STEP 2) diff = 0.1

diff=1 # ex) 1 ==> 0.1 | 01 ==> 0.01 
old_diff=4

for i in {182..231} ; do
cd task_subset-$i/
mkdir re-0$diff
cp subset-$i.traj train.py re-0$diff/
echo "  "
echo "  ##### Training the model-$i using the subset-$i ... ##### "
echo "  ***  ediff / fdiff / ediff_tot = 0.$diff  *** " 
echo "  "
cd re-0$diff/

cat >train.py <<!
from theforce.calculator.active import ActiveCalculator, FilterDeltas
from theforce.util.parallel import mpi_init
from ase import Atoms
from ase.io import read, write, Trajectory
import glob
import os

common = dict(ediff=0.$diff, fdiff=0.$diff, ediff_tot=0.$diff, process_group=mpi_init())
kernel_kw = dict(lmax=3, nmax=3, exponent=4, cutoff=4.0)
if type($old_diff) == str:
    if glob.glob(f"../model-$i-0$old_diff.pckl") == []:
        covariance = None
    else:
        covariance = '../model-$i-0$old_diff.pckl'
else:
    covariance = None
ML_calc = ActiveCalculator(covariance=covariance,
                           calculator=None,
                           logfile='active-$i-0$diff.log',
                           pckl='model-$i-0$diff.pckl/',
                           tape='model-$i-0$diff.sgpr',
                           kernel_kw=kernel_kw,
                           **common)
data = Trajectory('subset-$i.traj')

ML_calc.include_data(data)
!
mpirun -np 8 python train.py

cd ../../
done

echo "  ##### Predicting using validation set through BCM ... ##### "

cat >bcm_score.py <<!
import os
import sys
import glob
import time
import numpy as np
from ase import Atoms, Atom
from ase.io import read, write, Trajectory
from theforce.util.parallel import mpi_init 
from nonactive_bcm_4score import *

diff = '0$diff' # 0.4 => 04 | 0.01 => 001
file_list = glob.glob("4_GB/subset-*.traj")
total_files_num = len(file_list)
kernel_model_dict = {}
for i in range(total_files_num):
    search_directory = os.getcwd()
    path = glob.glob(os.path.join(search_directory, "**", f'model-{i+182}-{diff}.pckl'), recursive=True)   
    print(path)
    if path != []:
        kernel_model_dict[f'key{i+1}'] = path[0]
process_group = mpi_init()
bcm_calc = BCMCalculator(process_group=process_group,
                         kernel_model_dict=kernel_model_dict)
rank = 0
if distrib.is_initialized():
   rank = distrib.get_rank ()

atoms = read('v_Si_N.traj', index=slice(None))

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
!
mpirun -np 8 python bcm_score.py

mkdir bcm-0$diff
cp bcm_score.py bcm-0$diff/
mv predict.log bcm-0$diff/



## STEP 3) diff = 0.01

diff=01 # ex) 1 ==> 0.1 | 01 ==> 0.01 
old_diff=1

for i in {182..231} ; do
cd task_subset-$i/re-0$old_diff/
mkdir re-0$diff
cp subset-$i.traj train.py re-0$diff/
echo "  "
echo "  ##### Training the model-$i using the subset-$i ... ##### "
echo "  ***  ediff / fdiff / ediff_tot = 0.$diff  *** " 
echo "  "
cd re-0$diff/

cat >train.py <<!
from theforce.calculator.active import ActiveCalculator, FilterDeltas
from theforce.util.parallel import mpi_init
from ase import Atoms
from ase.io import read, write, Trajectory
import glob
import os

common = dict(ediff=0.$diff, fdiff=0.$diff, ediff_tot=0.$diff, process_group=mpi_init())
kernel_kw = dict(lmax=3, nmax=3, exponent=4, cutoff=4.0)
if type($old_diff) == str:
    if glob.glob(f"../model-$i-0$old_diff.pckl") == []:
        covariance = None
    else:
        covariance = '../model-$i-0$old_diff.pckl'
else:
    covariance = None
ML_calc = ActiveCalculator(covariance=covariance,
                           calculator=None,
                           logfile='active-$i-0$diff.log',
                           pckl='model-$i-0$diff.pckl/',
                           tape='model-$i-0$diff.sgpr',
                           kernel_kw=kernel_kw,
                           **common)
data = Trajectory('subset-$i.traj')

ML_calc.include_data(data)
!
mpirun -np 8 python train.py

cd ../../../
done

echo "  ##### Predicting using validation set through BCM ... ##### "

cat >bcm_score.py <<!
import os
import sys
import glob
import time
import numpy as np
from ase import Atoms, Atom
from ase.io import read, write, Trajectory
from theforce.util.parallel import mpi_init 
from nonactive_bcm_4score import *

diff = '0$diff' # 0.4 => 04 | 0.01 => 001
file_list = glob.glob("4_GB/subset-*.traj")
total_files_num = len(file_list)
kernel_model_dict = {}
for i in range(total_files_num):
    search_directory = os.getcwd()
    path = glob.glob(os.path.join(search_directory, "**", f'model-{i+182}-{diff}.pckl'), recursive=True)   
    print(path)
    if path != []:
        kernel_model_dict[f'key{i+1}'] = path[0]
process_group = mpi_init()
bcm_calc = BCMCalculator(process_group=process_group,
                         kernel_model_dict=kernel_model_dict)
rank = 0
if distrib.is_initialized():
   rank = distrib.get_rank ()

atoms = read('v_Si_N.traj', index=slice(None))

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
!
mpirun -np 8 python bcm_score.py

mkdir bcm-0$diff
cp bcm_score.py bcm-0$diff/
mv predict.log bcm-0$diff/



