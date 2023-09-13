import numpy as np
import random
import os
from ase import Atoms, Atom
from ase.io import read, write, Trajectory
from mpi4py import MPI

def randomize_list(input_number):
    original_list = list(range(input_number))
    random.shuffle(original_list)
    randomized_list = original_list
    return randomized_list

def get_list_to_divide(filename, each_subset_size):
    """
    filename : Trajectory or xyz type
    """
    train = read(filename, index=slice(None))
    len_train = len(train)
    each_subset_size_list = []
    quotient = len_train // each_subset_size
    remainder = len_train % each_subset_size
    if remainder < (each_subset_size*0.9):
        quo = remainder // quotient
        rem = remainder % quotient
        for i in range(quotient):
            if i < rem:
                each_subset_size_list += [each_subset_size+quo+1]
            else:
                each_subset_size_list += [each_subset_size+quo]
    else:
        for i in range(quotient):
            each_subset_size_list += [each_subset_size]
        each_subset_size_list += [remainder]
    print(f'\n *** Each subset size list ***\n{each_subset_size_list}')
    print(f'\n sum(each_subset_size_list):{sum(each_subset_size_list)} == len(train):{len(train)}')
    print(f'\n """ \n This training data set will be randomly divided into {len(each_subset_size_list)}.')
    print(f' And models for each subset will be trained automatically. \n """ \n')
    return each_subset_size_list

def generate_subset_randomly(filename, each_subset_size=50):
    each_subset_size_list = get_list_to_divide(filename, each_subset_size)
    train = read(filename, index=slice(None))
    randomized_list = randomize_list(len(train))
    size = 0
    atoms = train.copy()
    for i in range(len(each_subset_size_list)):
        traj = Trajectory(filename=f'subset-{i}.traj', mode='w')
        for j in range(each_subset_size_list[i]):
            traj.write(atoms[randomized_list[j+size]])
        traj.close()
        size += each_subset_size_list[i]



filename = 'Si.traj'
generate_subset_randomly(filename, each_subset_size=50)
