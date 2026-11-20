from dataclasses import dataclass
from typing import Optional

import yaml

_MISSING = object()


def _get(d, path, default=_MISSING):
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            if default is _MISSING:
                raise SystemExit(f"config.yaml: missing mandatory key : '{path}'")
            return default
        cur = cur[key]
    return cur


@dataclass(frozen=True)
class Config:
    name: str
    seed: Optional[int]
    n_runs: int
    interactive: bool
    output_dir: Optional[str]
    n_molecules: int
    n_atoms: int
    temperature: str          
    density: float
    lammps_exe: Optional[str]
    n_procs: int
    max_jobs: int

    @property
    def dim(self):            
        return 3 * self.n_atoms * self.n_molecules


def load_config(path):
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    model_type = _get(raw, "model.type")
    if model_type != "kremer_grest":
        raise SystemExit(f"model.type='{model_type}' not yet implemented (only 'kremer_grest' model).")

    seed = _get(raw, "run.seed", None)
    cfg = Config(
        name=_get(raw, "run.name", "nald"),
        seed=None if seed is None else int(seed),
        n_runs=int(_get(raw, "run.n_runs", 1)),
        interactive=bool(_get(raw, "run.interactive", False)),
        output_dir=_get(raw, "run.output_dir", None),
        n_molecules=int(_get(raw, "model.kremer_grest.n_molecules")),
        n_atoms=int(_get(raw, "model.kremer_grest.n_atoms")),
        temperature=str(_get(raw, "model.kremer_grest.temperature")),
        density=float(_get(raw, "model.kremer_grest.density", 0.1)),
        lammps_exe=_get(raw, "lammps.executable", None),
        n_procs=int(_get(raw, "executor.n_procs", 1)),
        max_jobs=int(_get(raw, "executor.max_parallel_jobs", 1)),
    )
    for field in ("n_molecules", "n_atoms", "n_runs", "n_procs", "max_jobs"):
        if getattr(cfg, field) < 1:
            raise SystemExit(f"config.yaml: '{field}' must be >= 1")
    if cfg.density <= 0:
        raise SystemExit("config.yaml: 'density' must be > 0")
    return cfg, raw