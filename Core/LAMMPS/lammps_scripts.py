from LAMMPS.lammps_runner import run_lammps
from LAMMPS.convert_data_bond import convert
import os

STEPS = [
    ("Balancing the system", "scripts/in.molecules_N"),
    ("Cooling the system", "scripts/in.KG_cool"),
    ("Computing the Hessian", "scripts/in.KG_cool_Hessian"),
    ("Computing affine forces", "scripts/in.KG_cool_affine"),
]

def main(N_atoms, N_molecules, T, id, n_procs=1, threads=1, exe=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dir_name = os.path.abspath(os.path.join(script_dir, f'../../Data/run_{id}_{T}_{N_atoms}_{N_molecules}'))
    convert(os.path.join(dir_name, 'atoms.csv'),
            os.path.join(dir_name, 'molecules_bonds.data'), N_atoms, N_molecules)

    variables = dict(T=T, id=id, N_atoms=N_atoms, N_molecules=N_molecules, dir_name=dir_name)
    for label, rel in STEPS:
        print(f"[run {id}] {label}...", flush=True)
        run_lammps(os.path.join(script_dir, rel), variables, n_procs, threads, exe,
                   cwd=dir_name, log=os.path.join(dir_name, f"log.{os.path.basename(rel)}.lammps"))