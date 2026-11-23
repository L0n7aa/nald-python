"""python -m nald.cli run config.yaml     (ou: nald run config.yaml après pip install -e .)"""
import argparse
import dataclasses
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import yaml

from nald.config import load_config
from nald.structure.paths import create_run_dir


def ask(question):
    while True:
        r = input(question + " [y/n] :").strip().lower()
        if r in ("y", "n"):
            return r == "y"
        print("Incorrect response.")


def git_state():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=here,
                                         stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=here,
                                             stderr=subprocess.DEVNULL, text=True).strip())
        return commit, dirty
    except Exception:
        return None, None


def save_provenance(paths, raw, cfg, index):
    info = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_state(),
        "python": sys.version.split()[0],
        "run_index": index,
        "seed_used": cfg.seed,
        "config": raw,
    }
    with open(os.path.join(paths.run_dir, "config.resolved.yaml"), "w") as f:
        yaml.safe_dump(info, f, sort_keys=False)


def run_single(cfg, raw, index=0):
    if cfg.seed is not None:
        cfg = dataclasses.replace(cfg, seed=cfg.seed + index)

    if not cfg.interactive:
        import matplotlib
        matplotlib.use("Agg")      # doit être choisi AVANT le premier import de pyplot

    from nald.structure.structure import structure
    from nald.lammps.lammps_scripts import main as lmp_scripts
    from nald.diagonalization.scripts_diag import main as diag
    from nald.modulus.modulus import compute as mod

    paths = create_run_dir(cfg)

    attempt = 0
    while True:
        structure(cfg, paths, show=cfg.interactive, attempt=attempt)
        if not cfg.interactive or not ask("Do you want to regenerate the system?"):
            break
        print("Regenerating...")
        attempt += 1

    save_provenance(paths, raw, cfg, index)
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
    r = sub.add_parser("run", help="lancer un calcul à partir d'un fichier YAML")
    r.add_argument("config")
    r.add_argument("--index", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    config_path = os.path.abspath(args.config)
    cfg, raw = load_config(config_path)
    if args.index is not None or cfg.n_runs == 1:
        run_single(cfg, raw, args.index or 0)
    else:
        run_batch(cfg, config_path)


if __name__ == "__main__":
    main()