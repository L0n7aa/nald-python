import numpy as np


def load_atoms(path):
    print("Loading data...")
    atoms_data = np.loadtxt(path, delimiter=",", skiprows=1)
    mol_atoms = {}
    for mol_id in np.unique(atoms_data[:, 0].astype(int)):
        mol_data = atoms_data[atoms_data[:, 0] == mol_id]
        mol_atoms[int(mol_id)] = mol_data[mol_data[:, 1].argsort()]
    return atoms_data, mol_atoms

def compute_box(atoms_data, margin=1e-6):
    mins = atoms_data[:, 2:5].min(axis=0) - margin
    maxs = atoms_data[:, 2:5].max(axis=0) + margin
    return mins, maxs

def build_atom_and_bond_lines(mol_atoms, mass_type, mol_id_offset):
    atom_lines = []
    bond_lines = []
    global_id = 1
    bond_id = 1
    for mol_id in sorted(mol_atoms):
        mol_data = mol_atoms[mol_id]
        ids_mol = []
        for row in mol_data:
            x, y, z = row[2], row[3], row[4]
            atom_lines.append(
                f"{global_id} {mol_id + mol_id_offset} {mass_type} "
                f"{x:.10f} {y:.10f} {z:.10f}\n"
            )
            ids_mol.append(global_id)
            global_id += 1
        for i in range(len(ids_mol) - 1):
            bond_lines.append(f"{bond_id} 1 {ids_mol[i]} {ids_mol[i+1]}\n")
            bond_id += 1
    return atom_lines, bond_lines


def check_counts(atom_lines, bond_lines, N_total_atoms, expected_bonds):
    if N_total_atoms is not None and len(atom_lines) != N_total_atoms:
        raise RuntimeError(f"Expected {N_total_atoms} atoms, found {len(atom_lines)}")
    if expected_bonds is not None and len(bond_lines) != expected_bonds:
        raise RuntimeError(f"Expected {expected_bonds} bonds, found {len(bond_lines)}")


def write_data_file(path, atom_lines, bond_lines, mins, maxs, mass_type, mass_value):
    print(f"Writing {path}...")
    with open(path, "w") as f:
        f.write("LAMMPS data file converted for atom_style bond\n\n")
        f.write(f"{len(atom_lines)} atoms\n")
        f.write("1 atom types\n")
        f.write(f"{len(bond_lines)} bonds\n")
        f.write("1 bond types\n\n")
        f.write(f"{mins[0]:.10f} {maxs[0]:.10f} xlo xhi\n")
        f.write(f"{mins[1]:.10f} {maxs[1]:.10f} ylo yhi\n")
        f.write(f"{mins[2]:.10f} {maxs[2]:.10f} zlo zhi\n")
        f.write("\nMasses\n\n")
        f.write(f"{mass_type} {mass_value}\n")
        f.write("\nAtoms # bond\n\n")
        f.writelines(atom_lines)
        f.write("\nBonds\n\n")
        f.writelines(bond_lines)


# Main function
def convert(atoms_file, output_file, N_atoms, N_molecules):
    mass_type = 1
    mass_value = 1.0
    mol_id_offset = 1 
    N_total_atoms = N_atoms*N_molecules
    expected_bonds = (N_atoms - 1)*N_molecules    
    atoms_data, mol_atoms = load_atoms(atoms_file)
    mins, maxs = compute_box(atoms_data)
    atom_lines, bond_lines = build_atom_and_bond_lines(mol_atoms, mass_type, mol_id_offset)
    check_counts(atom_lines, bond_lines, N_total_atoms, expected_bonds)
    write_data_file(output_file, atom_lines, bond_lines, mins, maxs, mass_type, mass_value)