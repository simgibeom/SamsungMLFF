
for i in 16 27 32 64 214 510 ; do
cd bcm_sub-$i/bcm_sub-$i-01/bcm_sub-$i-001/
#rm -r bcm_sub-$i-001
mkdir bcm_sub-$i-0001
cp Si_$i.traj v_Si_$i.traj mpi.sh train.py bcm_sub-$i-0001/
cd bcm_sub-$i-0001/

for PBS_O_WORKDIR in '$PBS_O_WORKDIR' ; do
cat >mpi.sh <<!
#!/bin/sh
#PBS -V
#PBS -N Si-$i-0001-train
#PBS -q flat 
#PBS -A vasp
#PBS -l select=2:ncpus=32:mpiprocs=32:ompthreads=1
#PBS -l walltime=48:00:00

cd $PBS_O_WORKDIR
mpirun -np 64 python train.py
!
done

cat >train.py <<!
from theforce.calculator.active import ActiveCalculator, FilterDeltas
from theforce.util.parallel import mpi_init
from ase import Atoms
from ase.io import read, write, Trajectory

common = dict(ediff=0.001, fdiff=0.001, ediff_tot=0.001, process_group=mpi_init())
kernel_kw = dict(lmax=3, nmax=3, exponent=4, cutoff=6.0)

ML_calc = ActiveCalculator(covariance='../model-$i-001.pckl',
                           calculator=None,
                           logfile='active-$i-0001.log',
                           pckl='model-$i-0001.pckl/',
                           tape='model-$i-0001.sgpr',
                           kernel_kw=kernel_kw,
                           **common)
data = Trajectory('Si_$i.traj')

ML_calc.include_data(data)
!

qsub mpi.sh
cd ../../../../
done
