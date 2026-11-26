import os
from dataclasses import dataclass
from typing import Optional, Tuple

import yaml

MODEL_TYPES = ("kremer_grest", "atomistic")
ALL_CHANNELS = ("xx", "yy", "zz", "xy", "xz", "yz")

STYLE_CMDS = {"bond_style", "angle_style", "dihedral_style", "improper_style", "pair_style",
              "pair_modify", "special_bonds", "kspace_style", "kspace_modify", "dielectric",
              "neighbor", "neigh_modify", "comm_modify"}
COEFF_CMDS = {"mass", "pair_coeff", "bond_coeff", "angle_coeff",
              "dihedral_coeff", "improper_coeff"}

KG_PRESET = {
    "units": "lj",
    "atom_style": "bond",
    "forcefield.styles": ["bond_style fene", "special_bonds fene",
                          "pair_style lj/cut 2.5", "pair_modify shift yes"],
    "forcefield.coefficients": ["mass * 1.0", "bond_coeff 1 30.0 1.5 1.0 1.0",
                                "pair_coeff * * 1.0 1.0 2.5"],
}
_MISSING = object()


@dataclass(frozen=True)
class Config:
    # execution
    name: str
    seed: Optional[int]
    n_runs: int
    interactive: bool
    output_dir: Optional[str]
    keep_scripts: bool
    lammps_exe: Optional[str]
    n_procs: int
    max_jobs: int
    # model
    model_type: str
    equilibrated: bool
    generate: bool
    temperature: str
    n_molecules: int
    n_atoms: int
    density: float
    # LAMMPS system
    data_file: Optional[str]
    units: str
    atom_style: str
    styles: Tuple[str, ...]
    coefficients: Tuple[str, ...]
    # NALD parameters
    hessian_style: str
    hessian_displacement: float
    hessian_binary: bool
    delta: float
    channels: Tuple[str, ...]

    @property
    def dim(self):
        return 3 * self.n_atoms * self.n_molecules


def _commands(lines, allowed, label):
    if not isinstance(lines, (list, tuple)):
        raise SystemExit(f"{label} must be a list of LAMMPS command")
    out = []
    for n, raw in enumerate(lines, 1):
        line = str(raw).split("#", 1)[0].strip()
        if not line:
            continue
        if any(c in line for c in "$`\\&;|<>") or line.split()[0] not in allowed:
            raise SystemExit(f"{label}, entry {n}: '{line}' denied "
                             f"(commands authorized: {', '.join(sorted(allowed))})")
        out.append(line)
    return tuple(out)


def _leaves(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict) and v:
            yield from _leaves(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}"


def load_config(path):
    """Return (Config, dict brut du YAML)."""
    path = os.path.abspath(path)
    base = os.path.dirname(path)
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    used = set()

    def get(key, default=_MISSING):
        used.add(key)
        cur = raw
        for k in key.split("."):
            if not isinstance(cur, dict) or k not in cur:
                if default is _MISSING:
                    raise SystemExit(f"config.yaml: missing mandatory key: '{key}'")
                return default
            cur = cur[k]
        return cur

    def resolve(p):
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))

    mtype = get("model.type")
    if mtype not in MODEL_TYPES:
        raise SystemExit(f"model.type='{mtype}' unknown (valid: {', '.join(MODEL_TYPES)})")

    eq = bool(get("model.equilibrated", False))
    generate = bool(get("model.generate", mtype == "kremer_grest" and not eq))
    if mtype == "atomistic" and generate:
        raise SystemExit("model.type 'atomistic' required 'generate: false' "
                         "(nald does not construct atomistic systems for now).")

    preset = KG_PRESET if mtype == "kremer_grest" else {}

    def sysget(key):
        return get(f"model.{key}", preset.get(key, _MISSING))

    seed, out_dir = get("run.seed", None), get("run.output_dir", None)
    channels = tuple(get("nald.affine.channels", list(ALL_CHANNELS)))
    raw_data = get("model.data_file", None if generate else _MISSING)

    cfg = Config(
        name=str(get("run.name", "nald")),
        seed=None if seed is None else int(seed),
        n_runs=int(get("run.n_runs", 1)),
        interactive=bool(get("run.interactive", False)),
        output_dir=None if out_dir is None else resolve(str(out_dir)),
        keep_scripts=bool(get("output.keep_scripts", False)),
        lammps_exe=get("lammps.executable", None),
        n_procs=int(get("executor.n_procs", 1)),
        max_jobs=int(get("executor.max_parallel_jobs", 1)),
        model_type=mtype,
        equilibrated=eq,
        generate=generate,
        temperature=str(get("model.temperature")),
        n_molecules=int(get("model.n_molecules", 1)),
        n_atoms=int(get("model.n_atoms")),
        density=float(get("model.density", 0.1)),
        data_file=None if generate or raw_data is None else resolve(str(raw_data)),
        units=str(sysget("units")),
        atom_style=str(sysget("atom_style")),
        styles=_commands(sysget("forcefield.styles"), STYLE_CMDS, "forcefield.styles"),
        coefficients=_commands(get("model.forcefield.coefficients",
                                   preset.get("forcefield.coefficients", [])),
                               COEFF_CMDS, "forcefield.coefficients"),
        hessian_style=str(get("nald.hessian.style", "regular")),
        hessian_displacement=float(get("nald.hessian.displacement", 1e-8)),
        hessian_binary=bool(get("nald.hessian.binary", False)),
        delta=float(get("nald.affine.delta", 1e-5)),
        channels=channels,
    )

    if cfg.data_file and (not os.path.isfile(cfg.data_file) or " " in os.path.basename(cfg.data_file)):
        raise SystemExit(f"data_file introuvable (ou nom avec espace): {cfg.data_file}")
    if cfg.hessian_style not in ("regular", "eskm"):
        raise SystemExit("nald.hessian.style doit valoir 'regular' ou 'eskm'")
    if not channels or any(c not in ALL_CHANNELS for c in channels):
        raise SystemExit(f"nald.affine.channels invalide (valides: {ALL_CHANNELS})")
    if min(cfg.n_runs, cfg.n_procs, cfg.max_jobs, cfg.n_molecules, cfg.n_atoms) < 1 \
            or cfg.delta <= 0 or cfg.hessian_displacement <= 0 or cfg.density <= 0:
        raise SystemExit("config.yaml: entiers, densité, delta et displacement doivent être > 0")

    unused = [p for p in _leaves(raw) if p not in used]
    if unused:
        print("config.yaml: clés non utilisées par cette version:", ", ".join(unused))
    if not cfg.generate and cfg.n_runs > 1:
        print("config.yaml: n_runs est ignoré quand generate: false (le calcul est déterministe).")
    return cfg, raw