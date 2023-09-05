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

import csv
import pandas as pd


process_group = mpi_init()
common = dict(ediff=0.1, fdiff=0.1, process_group=process_group)
kernel_kw = dict(lmax=3, nmax=3, exponent=4, cutoff=6.0)
test = read('test.xyz',index=':')

calc = ActiveCalculator(covariance='model3.pckl',
                        calculator=None,
                        logfile='active.log4',
                        pckl='model.pckl/',
                        tape='model.sgpr',
                        kernel_kw=kernel_kw,
                        **common)
data = []
ID = []
energy = []
force = []
for i in range(len(test)):
	atom = test[i].copy()
	atom.calc = calc
	ID += [f"TEST_{i:04}"]
	energy += [atom.get_potential_energy()]
	force += [atom.get_forces()]
	print(f' # ### TEST_{i:04} ... ####')
data = {'ID': ID, 'energy': energy, 'force': force}
csv_file_path = "sample_submission_SGB.csv"

df = pd.DataFrame(data)
df['force'] = df['force'].apply(lambda x: np.array2string(x, precision=6, threshold=np.inf, separator=' '))
df.to_csv(csv_file_path,  index=False)
