import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import eigs


def load_hessian(hessian_filepath):
    hessian = np.load(hessian_filepath)
    return hessian

def diagonalizing(hessian, eigenvalues_filepath, eigenvectors_filepath):
    eigenvalues, eigenvectors = eigh(hessian, overwrite_a=True, check_finite=False, driver="evr")
    np.save(eigenvalues_filepath, eigenvalues)
    #np.save(eigenvectors_filepath, eigenvectors)
    return [eigenvalues, eigenvectors]

def gamma(eigenvalues, eigenvectors, gamma_eigen_filepath, gamma_data_filepath, AF_filepath, N_molecules, N_atoms, dim):
    N = N_molecules*N_atoms
    dataAF = np.genfromtxt(AF_filepath, skip_header=9,usecols=(1, 2, 3))
    AF = dataAF[:N].reshape(dim)
    gamma = np.dot(eigenvectors.T, AF)**2
    np.save(gamma_data_filepath, gamma)
    combined = np.column_stack((eigenvalues, gamma))
    np.save(gamma_eigen_filepath, combined)