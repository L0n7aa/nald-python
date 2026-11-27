import os
from datetime import datetime
import re
import subprocess
import sys
import yaml
import shutil

from NALD.nald.engine.lammps_runner import run_lammps
from NALD.nald.engine.datafile import check_consistency, describe, read_header
from NALD.nald.engine.scripts import build_scripts, commands_sh, make_jobs
from ..utils import git_state, sha256_file, sha256_text

STEPS = [
    ("Balancing the system", "scripts/in.molecules_N"),
    ("Cooling the system", "scripts/in.KG_cool"),
    ("Computing the Hessian", "scripts/in.KG_cool_Hessian"),
    ("Computing affine forces", "scripts/in.KG_cool_affine"),
]

def lammps_version(exe):
    try:
        from NALD.nald.engine.lammps_runner import find_lammps
        out = subprocess.run([find_lammps(exe), "-h"], capture_output=True, text=True, timeout=30).stdout
        m = re.search(r"Large-scale Atomic/Molecular Massively Parallel Simulator\s*-\s*(.+)", out)
        return m.group(1).strip() if m else None
    except Exception:
        return None

def save_provenance(run_dir, cfg, raw, **extra):
    """config.resolved.yaml: recproductibility for the calculation."""
    commit, dirty = git_state()
    info = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit, "git_dirty": dirty,
        "python": sys.version.split()[0],
        "lammps_version": lammps_version(cfg.lammps_exe),
        "model": cfg.model_type, "generate": cfg.generate,
        "seed_used": cfg.seed,
        **extra,
        "config": raw,
    }
    with open(os.path.join(run_dir, "config.resolved.yaml"), "w") as f:
        yaml.safe_dump(info, f, sort_keys=False)

def prepare(cfg, raw, run_dir, hdr=None):
    """Write LAMMPS scripts."""
    if hdr is None:
        hdr = read_header(cfg.data_file)
    print(f"Data file: {cfg.data_file}\n{describe(hdr)}\n")
    for msg in check_consistency(hdr, cfg):
        print(f"WARNING: {msg}")

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
        data_file=cfg.data_file, data_sha256=sha256_file(cfg.data_file),
        n_atoms=hdr["counts"].get("atoms"),
        units=cfg.units, atom_style=cfg.atom_style, temperature=cfg.temperature,
        forcefield={"styles": list(cfg.styles), "coefficients": list(cfg.coefficients)},
        hessian={"style": cfg.hessian_style, "displacement": cfg.hessian_displacement},
        affine={"scheme": "centered", "delta": cfg.delta, "channels": list(cfg.channels)},
        scripts_sha256={name: sha256_text(text) for name, text in scripts.items()},
    )
    print(f"\nScripts written in {inputs}")
    print(f"To run them by hand:  sh {os.path.join(run_dir, 'commands.sh')}")
    return jobs


def main(cfg, raw, paths, hdr, threads=1):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    variables = dict(T=cfg.temperature, id=paths.run_id, N_atoms=cfg.n_atoms,
                     N_molecules=cfg.n_molecules, dir_name=paths.run_dir)
    for label, rel in STEPS:
        print(f"[run {paths.run_id}] {label}...", flush=True)
        run_lammps(os.path.join(script_dir, rel), variables, cfg.n_procs, threads, cfg.lammps_exe,
                   cwd=paths.run_dir,
                   log=os.path.join(paths.run_dir, f"log.{os.path.basename(rel)}.lammps"))
    jobs = prepare(cfg, raw, paths.run_dir, hdr=hdr)
    for job in jobs:
        print(f"[{job.name}] running...", flush=True)
        try:
            run_lammps(job.script, {}, cfg.n_procs, 1, cfg.lammps_exe, cwd=paths.run_dir, log=job.log)
        except Exception:
            print(f"Échec de '{job.name}': scripts conservés dans {os.path.join(paths.run_dir, 'inputs')}")
            raise
        if not cfg.keep_scripts:
            shutil.rmtree(os.path.join(paths.run_dir, "inputs"))
            os.remove(os.path.join(paths.run_dir, "commands.sh"))
        print(f"Done. LAMMPS outputs in {os.path.join(paths.run_dir, 'raw')}")