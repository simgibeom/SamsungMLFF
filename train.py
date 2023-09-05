from theforce.calculator.socketcalc import SocketCalculator
from theforce.calculator.active import ActiveCalculator, FilterDeltas
from theforce.similarity.sesoap import SeSoapKernel, SubSeSoapKernel
from theforce.util.parallel import mpi_init
from theforce.util.aseutil import init_velocities, make_cell_upper_triangular
from ase.build import bulk
from ase.md.npt import NPT 
from ase import units
from ase.calculators.vasp import Vasp

from ase import Atoms
from ase.io import read, write, Trajectory


ML_calc = ActiveCalculator(calculator=None,
	                       process_group=mpi_init(),
						   ediff=0.4)

data = Trajectory('Si.traj')

ML_calc.include_data(data)
