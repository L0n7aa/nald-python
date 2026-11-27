"""python -m nald.cli run config.yaml     (or: nald run config.yaml after pip install -e .)
"""

import argparse
import dataclasses
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
import matplotlib


from nald.config import load_config
from NALD.nald.engine.datafile import read_header
from nald.model.paths import create_run_dir


def run_single(cfg, raw, index=0):
    if cfg.seed is not None:
        cfg = dataclasses.replace(cfg, seed=cfg.seed + index)

    hdr = read_header(cfg.data_file) if not cfg.generate else None
    paths = create_run_dir(cfg, n_atoms=hdr["counts"].get("atoms") if hdr else None)

    from nald.model.structure import structure
    from NALD.nald.engine.lammps_scripts import main as lmp_scripts, save_provenance
    from nald.diagonalization.scripts_diag import main as diag
    from nald.modulus.modulus import compute as mod

    attempt = 0
    if cfg.generate:
        structure(cfg, paths, show=cfg.interactive, attempt=attempt)

    save_provenance(paths.run_dir, cfg, raw, run_index=index)

    lmp_scripts(cfg, paths)
    diag(cfg, paths)
    mod(paths)
    print(f"Done. Results in {paths.run_dir}")


def run_batch(cfg, config_path):
    jobs = min(cfg.max_jobs, cfg.n_runs)
    cpus = os.cpu_count() or 1
    if jobs * cfg.n_procs > cpus:
        print(f"Warning: {jobs * cfg.n_procs} cores requested but only {cpus} available.")
    os.makedirs("Logs", exist_ok=True)

    def one(i):
        cmd = [sys.executable, "-m", "nald.cli", "run", config_path, "--index", str(i)]
        env = {**os.environ, "OMP_NUM_THREADS": str(max(1, cpus // (jobs * cfg.n_procs)))}
        with open(f"Logs/run_{i}.log", "w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
        return i, rc

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for i, rc in ex.map(one, range(cfg.n_runs)):
            print(f"run {i} : {'OK' if rc == 0 else f'ERROR (code {rc}), see Logs/run_{i}.log'}")

def main(argv=None):
    ap = argparse.ArgumentParser(prog="nald")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="launch calculation from yaml file")
    r.add_argument("config")
    r.add_argument("--index", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    config_path = os.path.abspath(args.config)
    cfg, raw = load_config(config_path)

    index = getattr(args, "index", None)
    if index is not None or cfg.n_runs == 1:
        run_single(cfg, raw, index or 0)
    else:
        run_batch(cfg, config_path)


if __name__ == "__main__":
    matplotlib.use("Agg")
    main()