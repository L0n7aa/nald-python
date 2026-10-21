import numpy as np
import os
import config

from Diagonalization.hessian import organize, symetrizing
from Diagonalization.diagonalization import load_hessian, diagonalizing, gamma

# Parameters
N_atoms = config.N_atoms
N_molecules = config.N_molecules
T = config.T
id = config.id
dim = config.get_dim()

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
dir_name = os.path.join(script_dir + f'/../../Data/run_{id}_{T}_{N_atoms}_{N_molecules}')

old_hessian = os.path.join(dir_name, "Hessian.dat")
hessian_filepath = os.path.join(dir_name, "hessian.npy")

eigenvalues_filepath = os.path.join(dir_name, "eigenvalues.npy")
eigenvectors_filepath = os.path.join(dir_name, "eigenvectors.npy")
gamma_eigen_filepath = os.path.join(dir_name, "eigen_gamma.npy")
gamma_data_filepath = os.path.join(dir_name, "gamma.npy")
AF_filepath = os.path.join(dir_name, "AF.dat")

def get_hessian():
    print("Organizing and symetrizing the hessian...")
    organize(dim, old_hessian, hessian_filepath)
    symetrizing(dim, hessian_filepath)
    print("Data saved in hessian.npy successfully")

def diagonalization():
    print("Loading Hessian...")
    hessian = load_hessian(hessian_filepath)

    print("Diagonalizing...")
    [eigenvalues, eigenvectors] = diagonalizing(hessian, eigenvalues_filepath, eigenvectors_filepath)
    del hessian

    print("Calculations gamma...")
    gamma(eigenvalues, eigenvectors, gamma_eigen_filepath, gamma_data_filepath, AF_filepath, N_molecules, N_atoms, dim)

    print("Data saved successfully")

def main():
    get_hessian()
    diagonalization()