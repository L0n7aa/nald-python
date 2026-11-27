import numpy as np
import os
import nald.config as config
from dataclasses import dataclass

DEFAULT_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Data"))

_FILES = dict(
    atoms_csv="atoms.csv",
    molecules_npy="molecules_opt.npy",
    image="visualization.png",
    bonds_data="molecules_bonds.data",
    hessian_dat="Hessian.dat",
    hessian_npy="hessian.npy",
    eigenvalues="eigenvalues.npy",
    eigenvectors="eigenvectors.npy",
    eigen_gamma="eigen_gamma.npy",
    gamma="gamma.npy",
    af="AF.dat",
    volume="volume.dat",
    gp="Gp.npy",
    gdp="Gdp.npy",
)


@dataclass(frozen=True)
class RunPaths:
    run_dir: str
    run_id: int

    def __getattr__(self, name):           # paths.atoms_csv, paths.af, ...
        if name in _FILES:
            return os.path.join(self.run_dir, _FILES[name])
        raise AttributeError(name)


def create_run_dir(cfg):
    """Replace ini_opt.paths(): create Data/run_<id>_<T>_<N_atoms>_<N_molecules>."""
    data_dir = os.path.abspath(cfg.output_dir) if cfg.output_dir else DEFAULT_DATA_DIR
    os.makedirs(data_dir, exist_ok=True)
    run_id = len(os.listdir(data_dir))
    while True:
        run_dir = os.path.join(data_dir, f"run_{run_id}_{cfg.temperature}_{cfg.n_atoms}_{cfg.n_molecules}")
        try:
            os.makedirs(run_dir)
            break
        except FileExistsError:
            run_id += 1
    return RunPaths(run_dir, run_id)

