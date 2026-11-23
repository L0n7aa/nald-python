"python -m nald.verify  RUN_A  RUN_B  [--rtol 1e-6] [--atol 1e-10] [--only moduli]"

import argparse
import os
import sys

import numpy as np
import yaml

CHECKS = [
    ("generation", "atoms.csv", "bytes"),
    ("lammps", "volume.dat", "num"),
    ("lammps", "AF.dat", "af"),
    ("hessian", "hessian.npy", "npy"),
    ("diag", "eigenvalues.npy", "npy"),
    ("diag", "eigen_gamma.npy", "eigen_gamma"),
    ("moduli", "Gp.npy", "npy"),
    ("moduli", "Gdp.npy", "npy"),
]

OPTIONAL = {"hessian.npy"}

HINTS = {
    "generation": "different initial systems: check seed_used, density, n_atoms, and git commit.",
    "lammps": "same initial system but different dynamics: different number of MPI/thread cores, "
              "LAMMPS version, or machine (MD is chaotic).",
    "hessian": "same LAMMPS output but different Hessian: organize()/symmetrizing() or LAMMPS version.",
    "diag": "same Hessian but different spectrum: SciPy/LAPACK version or modified solver.",
    "moduli": "same modes but different G', G'': modulus.py or frequency parameters modified.",
}


def _load(path, kind):
    if kind == "bytes":
        with open(path, "rb") as f:
            return {"content": f.read()}
    if kind == "num":
        return {"value": np.atleast_1d(np.loadtxt(path))}
    if kind == "af":                                   
        return {"Xi": np.genfromtxt(path, skip_header=9, usecols=(1, 2, 3))}
    if kind == "npy":
        return {"array": np.load(path, mmap_mode="r")}
    if kind == "eigen_gamma":
        data = np.load(path)
        return {"eigenvalues": data[:, 0], "Gamma sum": np.array([data[:, 1].sum()])}
    raise ValueError(kind)


def _compare(a, b, rtol, atol):
    if isinstance(a, bytes):
        return a == b, 0.0 if a == b else float("nan"), 0.0 if a == b else float("nan")
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return False, float("nan"), float("nan")
    diff = np.abs(a - b)
    max_abs = float(diff.max()) if diff.size else 0.0
    max_rel = float((diff / np.maximum(np.abs(b), atol)).max()) if diff.size else 0.0
    return bool(np.allclose(a, b, rtol=rtol, atol=atol)), max_abs, max_rel


def _flatten(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def _read_prov(run_dir):
    path = os.path.join(run_dir, "config.resolved.yaml")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return yaml.safe_load(f) or {}


def check_provenance(dir_a, dir_b):
    """Compare parameters. True = identical, False = different, None = not comparable."""
    pa, pb = _read_prov(dir_a), _read_prov(dir_b)
    if pa is None or pb is None:
        print("  config.resolved.yaml missing in one of the directories (old run?): "
              "parameters not compared, only files are.\n")
        return None                                   # unknown: do not fail verification

    ca, cb = _flatten(pa.get("config")), _flatten(pb.get("config"))
    ok = True
    for key in sorted(set(ca) | set(cb)):
        if ca.get(key, "<absent>") != cb.get(key, "<absent>"):
            print(f"  different parameter  {key}: {ca.get(key, '<absent>')!r} vs {cb.get(key, '<absent>')!r}")
            ok = False

    if pa.get("seed_used") != pb.get("seed_used"):
        print(f"  different seed: {pa.get('seed_used')} vs {pb.get('seed_used')}")
        ok = False
    if pa.get("git_commit") != pb.get("git_commit"):
        print(f"  (info) different commits: {pa.get('git_commit')} vs {pb.get('git_commit')}")
    if ok:
        print("  parameters and seed identical")
    print()
    return ok


def verify(dir_a, dir_b, rtol=1e-6, atol=1e-10, only=None):
    for d in (dir_a, dir_b):
        if not os.path.isdir(d):
            print(f"Directory not found: {d}")
            return 2

    print(f"A: {dir_a}\nB: {dir_b}\ntolerances: rtol={rtol:g}, atol={atol:g}\n")
    print("Parameters:")
    config_ok = check_provenance(dir_a, dir_b)

    print(f"{'stage':<11}{'file':<18}{'quantity':<18}{'status':<7}{'abs diff':>12}{'rel diff':>12}")
    first_bad = None
    all_ok = True
    for stage, fname, kind in CHECKS:
        if only and stage not in only:
            continue
        pa, pb = os.path.join(dir_a, fname), os.path.join(dir_b, fname)
        ea, eb = os.path.exists(pa), os.path.exists(pb)
        if not ea and not eb:
            continue                                   
        if ea != eb and fname in OPTIONAL:
            print(f"{stage:<11}{fname:<18}{'-':<18}{'SKIP':<7}  optional file present in only one directory")
            continue
        if ea != eb:
            print(f"{stage:<11}{fname:<18}{'-':<18}{'FAIL':<7}  present in only one directory")
            all_ok, first_bad = False, first_bad or stage
            continue
        try:
            da, db = _load(pa, kind), _load(pb, kind)
        except Exception as exc:
            print(f"{stage:<11}{fname:<18}{'-':<18}{'FAIL':<7}  unable to read ({exc})")
            all_ok, first_bad = False, first_bad or stage
            continue
        for label in da:
            ok, max_abs, max_rel = _compare(da[label], db[label], rtol, atol)
            status = "OK" if ok else "FAIL"
            print(f"{stage:<11}{fname:<18}{label:<18}{status:<7}{max_abs:>12.3e}{max_rel:>12.3e}")
            if not ok:
                all_ok, first_bad = False, first_bad or stage

    print()
    if all_ok and config_ok is not False:
        print("REPRODUCIBLE: all files match within tolerances.")
        return 0
    if config_ok is False:
        print("WARNING: parameters differ, comparison is not conclusive.")
    if first_bad:
        print(f"NON-REPRODUCIBLE: first divergence at stage '{first_bad}'.")
        print(f"  -> {HINTS[first_bad]}")
    return 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nald verify", description=__doc__.split("\n")[0])
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-10)
    ap.add_argument("--only", default=None,
                    help="stages to compare, comma-separated "
                         "(generation,lammps,hessian,diag,moduli). E.g.: --only moduli")
    args = ap.parse_args(argv)
    only = set(args.only.split(",")) if args.only else None
    return verify(args.run_a, args.run_b, args.rtol, args.atol, only)


if __name__ == "__main__":
    sys.exit(main())