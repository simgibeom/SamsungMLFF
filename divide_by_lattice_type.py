from collections import Counter
import numpy as np

from ase import Atoms
from ase.visualize import view
from ase.io import read, write, Trajectory
from ase.constraints import FixAtoms 
from ase.data.pubchem import pubchem_atoms_search, pubchem_atoms_conformer_search 
from ase.geometry import get_distances
from ase.data import atomic_numbers, atomic_names, atomic_masses, covalent_radii
from ase.geometry import cell
from ase.build import make_supercell

import linecache
import sys
import math
import random
import os
import shutil

def bravais_lattice(lattice_constant, angle):
    """
    lattice_constant = [a, b, c]
    angle = [alpha, beta, gamma]
    """
    if 3 in Counter(lattice_constant).values():
        # a == b == c
        if Counter(angle)[90.0] == 3:
            # alpha == beta == gamma == 90.0
            type_of_lattice = 'Cubic'
        elif Counter(angle)[90.0] == 0 and 3 in Counter(angle).values():
            # alpha == beta == gamma != 90.0
            type_of_lattice = 'Rhombohedral'
        elif Counter(angle)[90.0] == 2:
            if Counter(angle)[120.0] == 1 or Counter(angle)[60.0] == 1:
                # alpha == beta == 90.0 and gamma == 120.0
                type_of_lattice = 'Hexagonal'
            else:
                type_of_lattice = 'Monoclinic'
        else:
            type_of_lattice = 'Triclinic'
    elif 2 in Counter(lattice_constant).values():
        # a == b != c
        if Counter(angle)[90.0] == 3:
            # alpha == beta == gamma == 90.0
            type_of_lattice = 'Tetragonal'
        elif Counter(angle)[90.0] == 2:
            if Counter(angle)[120.0] == 1 or Counter(angle)[60.0] == 1:
                # alpha == beta == 90.0 and gamma == 120.0
                type_of_lattice = 'Hexagonal'
            else:
                type_of_lattice = 'Monoclinic'
        else:
            type_of_lattice = 'Triclinic'
    else:
        if Counter(angle)[90.0] == 3:
            # alpha == beta == gamma == 90.0
            type_of_lattice = 'Orthorhombic'
        elif Counter(angle)[90.0] == 2:
            # alpha == gamma == 90.0 and beta != 90.0
            type_of_lattice = 'Monoclinic'
        else:
            # alpha != beta != gamma != 90
            type_of_lattice = 'Triclinic'
    return type_of_lattice

def find_angle_rad_2d(v1, v2):
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    angle_rad = np.arccos(np.dot(v1, v2) / (norm_v1*norm_v2))
    return angle_rad

def find_angle_deg_2d(v1, v2):
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    angle_deg = np.arccos(np.dot(v1, v2) / (norm_v1*norm_v2)) * (180/np.pi)
    return angle_deg

def find_lattice_type(cell):
    """
    cell : np.array of cell; like from "atoms.get_cell()"
    """
    v1 = cell[0]
    v2 = cell[1]
    v3 = cell[2]

    alpha = round(find_angle_deg_2d(v2, v3), 2)
    beta  = round(find_angle_deg_2d(v1, v3), 2)
    gamma = round(find_angle_deg_2d(v1, v2), 2)
    angle = [alpha, beta, gamma]

    a = round(np.linalg.norm(v1), 2)
    b = round(np.linalg.norm(v2), 2)
    c = round(np.linalg.norm(v3), 2)
    lattice_constant = [a, b, c]

    type_of_lattice = bravais_lattice(lattice_constant, angle)
    #print(f'\n##### Type of Bravais Lattice #####\n')
    #print(f'- Angle\n{angle}\n\n- Lattice constant\n{lattice_constant}\n')
    #print(f'==> Type : {type_of_lattice}')

    return type_of_lattice

def calculate_distances_to_point(point, atomic_positions):
    distances = []
    for pos in atomic_positions:
        distance = round(np.linalg.norm(point - pos), 3)
        distances.append(distance)
    return distances

def index_sort_by_energy(indices_list, energies_list):
    index_ene = {indices_list[i]:energies_list[i] for i in range(len(energies_list))}
    sorted_index = sorted(index_ene, key=index_ene.get)
    return sorted_index


if __name__ == "__main__":
    pwd = '.'
    atoms = Trajectory('Si_27.traj')
   # v_si = Trajectory('v_Si_16.traj')
    lattice_sort_indices = {}
    for i in range(len(atoms)):
        cell = atoms[i].get_cell().copy()
        lattice = find_lattice_type(cell)
        if lattice in lattice_sort_indices:
            lattice_sort_indices[lattice] += [i]
        else:
            lattice_sort_indices[lattice] = [i]
	for key in lattice_const_dict:
        value = lattice_const_dict[key]
        traj = Trajectory(f'Si_{key}.traj', 'w')
        print(key, len(value))
        for i in range(len(value)):
            traj.write(atoms[i])
        traj.close()
        
        os.makedirs(f'{pwd}/{key}', exist_ok = True)
        source_path  = os.path.join(pwd, f'Si_{key}.traj')
        destination_path = os.path.join(f'{pwd}/{key}', f'Si_{key}.traj')
        shutil.copy(source_path, destination_path)

        source_path  = os.path.join(pwd, 'random_divide.py')
        destination_path = os.path.join(f'{pwd}/{key}', 'random_divide.py')
        shutil.copy(source_path, destination_path)
        source_path  = os.path.join(pwd, 'auto_train.sh')
        destination_path = os.path.join(f'{pwd}/{key}', 'auto_train.sh')
        shutil.copy(source_path, destination_path)
