import argparse
import config
import os 
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-M', '--molecules',type=int, help='number of molecules')
    parser.add_argument('-A', '--atoms', type=int, help='number of atoms')
    parser.add_argument('-T', '--temperature', help='value of the temperature for this simulation')
    parser.add_argument('-n', '--np', type=int, default=1, help='MPI processes per simulation')
    parser.add_argument('-r', '--runs', type=int, default=1, help='number of simulations')
    parser.add_argument('-j', '--jobs', type=int, default=1, help='number of simulations running at the same time')
    parser.add_argument('--lammps', default=None, help='path to the LAMMPS executable')
    parser.add_argument('--yes', action='store_true', help='non-interactive mode (no prompts, no windows)')
    return parser.parse_args()

def ask(question):
    while True:
        r = input(question + ' [y/n] :').strip().lower()
        if r in ('y', 'n'):
            return r == 'y'
        print("Incorrect response.")

def launch_batch(args):
    jobs = min(args.jobs, args.runs)
    cpus = os.cpu_count() or 1
    if jobs * args.np > cpus:
        print(f"Warning: {jobs * args.np} cores requested but only {cpus} available.")
    os.makedirs("Logs", exist_ok=True)

    def one(i):
        cmd = [sys.executable, os.path.abspath(__file__),
               "-A", str(args.atoms), "-M", str(args.molecules), "-T", str(args.temperature),
               "-n", str(args.np), "--yes"]
        if args.lammps:
            cmd += ["--lammps", args.lammps]
        env = {**os.environ, "OMP_NUM_THREADS": str(max(1, cpus // (jobs * args.np)))}
        with open(f"Logs/run_{i}.log", "w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
        return i, rc

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for i, rc in ex.map(one, range(args.runs)):
            print(f"run {i} : {'OK' if rc == 0 else f'ERROR (code {rc}), see Logs/run_{i}.log'}")


def main():
    args = parse_args()

    if args.runs > 1:                   
        launch_batch(args)
        return

    config.N_atoms = args.atoms
    config.N_molecules = args.molecules
    config.T = args.temperature

    if args.yes:
        import matplotlib
        matplotlib.use("Agg")

    from NALD.Core.Structure.structure import structure
    from LAMMPS.lammps_scripts import main as lmp_scripts
    from Diagonalization.scripts_diag import main as diag
    from Modulus.modulus import compute as mod
    from Plot.plot import plot_moduli

    while True:
        structure(show=not args.yes)
        if args.yes or not ask("Do you want to regenerate the system?"):
            break
        print("Regenerating...")

    lmp_scripts(config.N_atoms, config.N_molecules, config.T, config.id,
                n_procs=args.np, exe=args.lammps)
    diag()
    mod()

# if __name__ == "__main__":
#     main() 