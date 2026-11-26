"""Construit les scripts LAMMPS à partir des scripts de base (templates/) et de la config."""
from dataclasses import dataclass

from nald.engine.render import render

SCALE_AXIS = {"xx": "x", "yy": "y", "zz": "z"}
TILT_LENGTH = {"xy": "ly", "xz": "lz", "yz": "lz"}      # déformation = tilt / longueur normale


@dataclass(frozen=True)
class Job:
    name: str
    script: str       # relatif au dossier du run
    log: str
    n_procs: int = 1


def _num(x):
    return format(x, ".16g")


def build_scripts(cfg, raw_dir="raw", system_in="inputs/system.in"):
    """Renvoie {nom_de_fichier: texte}. Aucune écriture: l'appelant décide quoi garder."""
    kspace = [l for l in cfg.styles if l.split()[0] in ("kspace_style", "kspace_modify")]

    blocks = []
    for i, ch in enumerate(cfg.channels):
        common = dict(CH=ch, RAW=raw_dir, FILEMODE="file" if i == 0 else "append")
        if ch in SCALE_AXIS:
            blocks.append(render("channel_scale.in", AXIS=SCALE_AXIS[ch], **common))
        else:
            blocks.append(render("channel_tilt.in", LEN=TILT_LENGTH[ch], **common))

    return {
        "system.in": render("system.in", UNITS=cfg.units, ATOM_STYLE=cfg.atom_style,
                            FORCEFIELD="\n".join(cfg.styles), DATA_FILE=cfg.data_file,
                            COEFFICIENTS="\n".join(cfg.coefficients)),
        "hessian.in": render("hessian.in", SYSTEM=system_in, RAW=raw_dir, STYLE=cfg.hessian_style,
                             DISPLACEMENT=_num(cfg.hessian_displacement),
                             BINARY=" binary yes" if cfg.hessian_binary else ""),
        "affine.in": render("affine.in", SYSTEM=system_in, DELTA=_num(cfg.delta),
                            KSPACE="\n".join(kspace), CHANNELS="\n".join(blocks)),
    }


def make_jobs(cfg):
    # le calcul affine est rapide: il passe d'abord pour que les erreurs apparaissent tôt
    return [Job("affine", "inputs/affine.in", "raw/log.affine", cfg.n_procs),
            Job("hessian", "inputs/hessian.in", "raw/log.hessian", cfg.n_procs)]


def commands_sh(cfg, jobs):
    exe = cfg.lammps_exe or "lmp"
    prefix = f"mpirun -np {cfg.n_procs} " if cfg.n_procs > 1 else ""
    lines = ["#!/bin/sh",
             "# Run from anywhere:  sh commands.sh      (LMP=/path/to/lmp to choose the executable)",
             'cd "$(dirname "$0")" || exit 1',
             f"LMP=${{LMP:-{exe}}}"]
    lines += [f"{prefix}$LMP -in {j.script} -log {j.log}" for j in jobs]
    return "\n".join(lines) + "\n"