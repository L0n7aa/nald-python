import os

from nald.lammps.convert_data_bond import convert
from nald.lammps.lammps_runner import run_lammps

STEPS = [
    ("Balancing the system", "scripts/in.molecules_N"),
    ("Cooling the system", "scripts/in.KG_cool"),
    ("Computing the Hessian", "scripts/in.KG_cool_Hessian"),
    ("Computing affine forces", "scripts/in.KG_cool_affine"),
]


def main(cfg, paths, threads=1):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    convert(paths.atoms_csv, paths.bonds_data, cfg.n_atoms, cfg.n_molecules)

    variables = dict(T=cfg.temperature, id=paths.run_id, N_atoms=cfg.n_atoms,
                     N_molecules=cfg.n_molecules, dir_name=paths.run_dir)
    for label, rel in STEPS:
        print(f"[run {paths.run_id}] {label}...", flush=True)
        run_lammps(os.path.join(script_dir, rel), variables, cfg.n_procs, threads, cfg.lammps_exe,
                   cwd=paths.run_dir,
                   log=os.path.join(paths.run_dir, f"log.{os.path.basename(rel)}.lammps"))