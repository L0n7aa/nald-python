from nald.diagonalization.hessian import organize, symetrizing
from nald.diagonalization.diagonalization import load_hessian, diagonalizing, gamma


def get_hessian(cfg, paths):
    print("Organizing and symetrizing the hessian...")
    organize(cfg.dim, paths.hessian_dat, paths.hessian_npy)
    symetrizing(cfg.dim, paths.hessian_npy)
    print("Data saved in hessian.npy successfully")

def diagonalization(cfg, paths):
    print("Loading Hessian...")
    hessian = load_hessian(paths.hessian_npy)

    print("Diagonalizing...")
    [eigenvalues, eigenvectors] = diagonalizing(hessian, paths.eigenvalues)
    del hessian

    print("Calculations gamma...")
    gamma(eigenvalues, eigenvectors, paths.eigen_gamma, paths.gamma, paths.af, cfg.n_molecules, cfg.n_atoms, cfg.dim)

    print("Data saved successfully")

def main(cfg, paths):
    get_hessian(cfg, paths)
    diagonalization(cfg,paths)