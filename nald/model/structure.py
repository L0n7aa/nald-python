import numpy as np

from nald.model.ini import SystemGenerator
from nald.model.show_data import load_data, save_data, visualization
from NALD.nald.model.convert_data_bond import convert


def structure(cfg, paths, attempt=0):
    if cfg.seed is None or attempt == 0:
        seed = cfg.seed                      
    else:
        seed = [cfg.seed, attempt]

    print("Generating optimized system...")
    molecules = SystemGenerator(cfg.n_atoms, cfg.n_molecules, cfg.density, seed).generate_system()
    if len(molecules) > 0:
        print(f"Success: {len(molecules)} molecules generated.")
        np.save(paths.molecules_npy, molecules)
        print("File saved as 'molecules_opt.npy'")
    else:
        print("Failure: Impossible to place even a single molecule.")

    molecules, n_molecules, n_atoms = load_data(paths.molecules_npy)
    print("Saving data in CSV files...")
    save_data(molecules, n_molecules, n_atoms, paths.atoms_csv)

    visualization(molecules, n_molecules, paths.image)
    convert(paths.atoms_csv, paths.bonds_data, cfg.n_atoms, cfg.n_molecules)