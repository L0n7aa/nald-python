import numpy as np
import config

from Situation.ini_opt import generate_system_optimized, paths
from Situation.show_data import load_data, save_data, visualization

npy_data_path, atom_path, image_path, config.id = paths()

def situation():
    print("Generating optimized system...")
    molecules = generate_system_optimized()
    if len(molecules) > 0:
        print(f"Success: {len(molecules)} molecules generated.")
        np.save(npy_data_path, molecules)
        print("File saved as 'molecules_opt.npy'")
    else:
        print("Failure: Impossible to place even a single molecule.")

    molecules, n_molecules, n_atoms = load_data(npy_data_path)
    print("Saving data in CSV files...")
    save_data(molecules, n_molecules, n_atoms, atom_path)

    visualization(molecules, n_molecules, image_path)
