"""python -m nald.cli run config.yaml     (or: nald run config.yaml after pip install -e .)

  nald run     config.yaml   -> calcul complet
  nald prepare config.yaml   -> (model.equilibrated: true) écrit les scripts LAMMPS sans les lancer

Deux chemins, un seul fichier:
  1. système généré (KG, equilibrated: false) : run_single / run_batch
  2. système fourni (equilibrated: true)      : prepare / run_equilibrated
Ils partagent git_state, lammps_version et save_provenance.
"""
import argparse
import dataclasses
import hashlib
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import yaml

from nald.config import load_config
from nald.engine.datafile import check_consistency, describe, read_header
from nald.engine.scripts import build_scripts, commands_sh, make_jobs
from nald.model.paths import create_run_dir


# ----------------------------------------------------------------- utilitaires communs
def ask(question):
    while True:
        r = input(question + " [y/n] :").strip().lower()
        if r in ("y", "n"):
            return r == "y"
        print("Incorrect response.")


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state():
    """(commit, dirty); dirty=True si des modifications ne sont pas commitées."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=here,
                                         stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=here,
                                             stderr=subprocess.DEVNULL, text=True).strip())
        return commit, dirty
    except Exception:
        return None, None


def lammps_version(exe):
    try:
        from nald.lammps.lammps_runner import find_lammps
        out = subprocess.run([find_lammps(exe), "-h"], capture_output=True, text=True, timeout=30).stdout
        m = re.search(r"Large-scale Atomic/Molecular Massively Parallel Simulator\s*-\s*(.+)", out)
        return m.group(1).strip() if m else None
    except Exception:
        return None


def save_provenance(run_dir, cfg, raw, **extra):
    """config.resolved.yaml: de quoi comprendre et refaire le calcul. `extra` = champs propres au chemin."""
    commit, dirty = git_state()
    info = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit, "git_dirty": dirty,
        "python": sys.version.split()[0],
        "lammps_version": lammps_version(cfg.lammps_exe),
        "model": cfg.model_type, "equilibrated": cfg.equilibrated,
        "seed_used": cfg.seed,
        **extra,
        "config": raw,
    }
    with open(os.path.join(run_dir, "config.resolved.yaml"), "w") as f:
        yaml.safe_dump(info, f, sort_keys=False)


# ----------------------------------------------------------------- chemin 2: système fourni
def equilibrated_run_dir(cfg, data_sha):
    base = os.path.join(cfg.output_dir or os.path.join(os.getcwd(), "results"), cfg.name)
    os.makedirs(base, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    digest = sha256_text(repr(cfg) + data_sha)[:6]
    run_dir, k = os.path.join(base, f"{stamp}_{digest}"), 0
    while True:
        try:
            os.makedirs(run_dir)
            return run_dir
        except FileExistsError:
            k += 1
            run_dir = os.path.join(base, f"{stamp}_{digest}_{k}")


def prepare(cfg, raw):
    """Crée le dossier du run, écrit les scripts (inputs/) et la provenance. Ne lance rien."""
    hdr = read_header(cfg.data_file)
    print(f"Data file: {cfg.data_file}\n{describe(hdr)}\n")
    for msg in check_consistency(hdr, cfg):
        print(f"WARNING: {msg}")

    data_sha = sha256_file(cfg.data_file)
    run_dir = equilibrated_run_dir(cfg, data_sha)
    inputs = os.path.join(run_dir, "inputs")
    os.makedirs(inputs)
    os.makedirs(os.path.join(run_dir, "raw"))

    scripts = build_scripts(cfg)
    for name, text in scripts.items():
        with open(os.path.join(inputs, name), "w") as f:
            f.write(text)
    jobs = make_jobs(cfg)
    with open(os.path.join(run_dir, "commands.sh"), "w") as f:
        f.write(commands_sh(cfg, jobs))

    save_provenance(
        run_dir, cfg, raw,
        data_file=cfg.data_file, data_sha256=data_sha, n_atoms=hdr["counts"].get("atoms"),
        units=cfg.units, atom_style=cfg.atom_style, temperature=cfg.temperature,
        forcefield={"styles": list(cfg.styles), "coefficients": list(cfg.coefficients)},
        hessian={"style": cfg.hessian_style, "displacement": cfg.hessian_displacement},
        affine={"scheme": "centered", "delta": cfg.delta, "channels": list(cfg.channels)},
        scripts_sha256={name: sha256_text(text) for name, text in scripts.items()},
    )
    print(f"\nScripts written in {inputs}")
    print(f"To run them by hand:  sh {os.path.join(run_dir, 'commands.sh')}")
    return run_dir, jobs

def run_single(cfg, raw, index=0):
    if cfg.equilibrated:
        run_dir, jobs = prepare(cfg, raw)
        from nald.lammps.lammps_runner import run_lammps
        for job in jobs:
            print(f"[{job.name}] running...", flush=True)
            try:
                run_lammps(job.script, {}, cfg.n_procs, 1, cfg.lammps_exe, cwd=run_dir, log=job.log)
            except Exception:
                print(f"Échec de '{job.name}': scripts conservés dans {os.path.join(run_dir, 'inputs')}")
                raise
        if not cfg.keep_scripts:
            shutil.rmtree(os.path.join(run_dir, "inputs"))
            os.remove(os.path.join(run_dir, "commands.sh"))
        print(f"Done. LAMMPS outputs in {os.path.join(run_dir, 'raw')}")
        return
    
    if cfg.seed is not None:
        cfg = dataclasses.replace(cfg, seed=cfg.seed + index)

    if not cfg.interactive:
        import matplotlib
        matplotlib.use("Agg")      

    from nald.model.structure import structure
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
    main()