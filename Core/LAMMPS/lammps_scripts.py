from lammps import lammps
import os
import config

from LAMMPS.convert_data_bond import convert

# Parameters 
N_atoms = config.N_atoms
N_molecules = config.N_molecules
T = config.T
id = config.id

script_dir = os.path.dirname(os.path.abspath(__file__))
dir_name = os.path.join(script_dir + f'/../../Data/run_{id}_{T}_{N_atoms}_{N_molecules}')

atoms_file = os.path.join(dir_name, 'atoms.csv')
output_file = os.path.join(dir_name, 'molecules_bonds.data')

molecules_path = os.path.join(script_dir, 'scripts/in.molecules_N')
hessian_path = os.path.join(script_dir, 'scripts/in.KG_cool_Hessian')
aff_path = os.path.join(script_dir, 'scripts/in.KG_cool_affine')


def lammps_instance(id, T, N_atoms, N_molecules):
    return lammps(cmdargs=["-var", "T", str(T), "-var", "id", str(id), "-var", "N_atoms", str(N_atoms), "-var", "N_molecules", str(N_molecules)])

def run_script(lmp, path):
    lmp.command("clear")
    lmp.command(f"variable dir_name string {dir_name}")
    lmp.file(path)

def main():
    print("Converting data for LAMMPS...")
    convert(atoms_file, output_file, N_atoms, N_molecules)
    print(f"File '{output_file}' successfully created.")

    lmp = lammps_instance(id, T, N_atoms, N_molecules)

    print("Balancing the system...")
    run_script(lmp, molecules_path)
    print("Files created : trajectory.lammpstrj, end.data")

    print("Calculations the Hessian...")
    run_script(lmp, hessian_path)
    print("Files created : Hessian.dat, end_Hessian.data")

    print("Calculations affine forces...")
    run_script(lmp, aff_path)
    print("Files created : AF.dat, end_affine.data")