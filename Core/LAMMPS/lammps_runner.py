import os
import shutil
import subprocess
import glob

CANDIDATES = ("lmp", "lmp_mpi", "lmp_serial", "lmp_linux", "lmp_mac", "lammps")


def find_lammps(explicit=None):
    for c in (explicit, os.environ.get("LAMMPS_EXE")):
        if c:
            p = shutil.which(c)
            if p:
                return p
            raise FileNotFoundError(f"LAMMPS introuvable : {c}")
    for name in CANDIDATES:
        p = shutil.which(name)
        if p:
            return p
    raise FileNotFoundError(
        "LAMMPS not found. Install LAMMPS with:\n"
        "  conda install -c conda-forge lammps\n"
        "or write directly its path with --lammps /chemin/lmp (or LAMMPS_EXE)."
    )

def _lib_dirs(exe):
    prefix = os.path.dirname(os.path.dirname(os.path.realpath(exe)))
    candidates = [os.environ.get("LAMMPS_LIB_DIR"),
                  os.path.join(prefix, "lib"),
                  os.path.join(prefix, "lib64")]
    return [d for d in candidates
            if d and glob.glob(os.path.join(d, "liblammps.so*"))]


def run_lammps(script, variables, n_procs=1, threads=1, exe=None, cwd=None, log=None):
    exe = find_lammps(exe)
    cmd = []
    if n_procs > 1:
        if "serial" in os.path.basename(exe):
            raise RuntimeError(f"{exe} is a serial version, the using of {n_procs} cores is impossible.")
        mpi = shutil.which("mpirun") or shutil.which("mpiexec")
        if not mpi:
            raise FileNotFoundError("mpirun/mpiexec not found, install OpenMPI or MPICH.")
        cmd += [mpi, "-np", str(n_procs)]
    cmd += [exe, "-in", script, "-log", log or "none"]
    if threads > 1:  
        cmd += ["-sf", "omp", "-pk", "omp", str(threads)]
    for k, v in variables.items():
        cmd += ["-var", k, str(v)]
    env = {**os.environ, "OMP_NUM_THREADS": str(threads)}
    libs = _lib_dirs(exe)
    if libs:
        old = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = os.pathsep.join(libs + ([old] if old else []))
    subprocess.run(cmd, check=True, cwd=cwd, env=env)