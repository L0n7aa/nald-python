import os
import numpy as np
from scipy.linalg import eigh


def load_hessian(hessian_filepath):
    mm = np.load(hessian_filepath, mmap_mode="r")
    return np.array(mm.T)  

def read_AF(AF_filepath, N):
    dataAF = np.genfromtxt(AF_filepath, skip_header=9, usecols=(1, 2, 3))
    return dataAF[:N].reshape(3 * N)

def chunk_path(chunk_dir, lo, hi, kind):
    return os.path.join(chunk_dir, f"{kind}_{lo:06d}_{hi:06d}.npy")

def diagonalizing_chunked(hessian_filepath, AF, chunk_dir, chunk=3000, save_vectors=False):
    """Diagonalization by blocks. For each blocks :
            - eigh(subset_by_index) with the 'evr' driver
            - gamma immediate projection = v.T @ AF
            - block is saved on disk (eigehnvalues + gamma)
        
    An already present block on disk is skipped. The Hessian is loaded from disk for each block, because calling the eigh() function with the option overwrite_a=True destroys it (it avoid a copy of 7.2 Go)."""
    os.makedirs(chunk_dir, exist_ok=True)
    n = AF.shape[0]

    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n) - 1
        out = chunk_path(chunk_dir, lo, hi, "eg")

        if os.path.exists(out):
            print(f"Modes {lo}-{hi} already computed, skipping it.", flush=True)
            continue

        print(f"Modes {lo}-{hi} : loading...", flush=True)
        H = load_hessian(hessian_filepath)

        print(f"Modes {lo}-{hi} : diagonalization...", flush=True)
        w, v = eigh(H, subset_by_index=[lo, hi], driver="evr",
                    overwrite_a=True, check_finite=False)
        del H

        g = (v.T @ AF)**2
        if save_vectors:
            np.save(chunk_path(chunk_dir, lo, hi, "vec"), v)
        del v

        tmp = out + ".tmp.npy"
        np.save(tmp, np.column_stack((w, g)))
        os.replace(tmp, out)
        print(f"Modes {lo}-{hi} : OK", flush=True)


def assemble(chunk_dir):
    """Assemble all blocks."""
    files = sorted(f for f in os.listdir(chunk_dir)
                   if f.startswith("eg_") and f.endswith(".npy")
                   and not f.endswith(".tmp.npy"))
    blocks = [np.load(os.path.join(chunk_dir, f)) for f in files]
    data = np.vstack(blocks)
    return data[:, 0], data[:, 1]  


if __name__ == "__main__":

    #Parameters
    N = 10000
    dim = 3 * N
    T = 1
    CHUNK = 3000 # 3 hours of Calculations     
    SAVE_VECTORS = False  

    # Directories and filepaths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    hessian_filepath = os.path.join(script_dir, "data", "hessian.npy")
    eigenvalues_filepath = os.path.join(script_dir, "data", "eigenvalues.npy")
    chunk_dir = os.path.join(script_dir, "data", "chunks")
    gamma_eigen_filepath = os.path.join(script_dir, "../../LAMMPS/output", f"eigen_gamma_T{T}.data")
    gamma_data_filepath = os.path.join(script_dir, "../../LAMMPS/output", f"gamma_T{T}.data")
    AF_filepath = os.path.join(script_dir, "../../LAMMPS/output", f"AF_T{T}.dat")

    # Main program
    print("Reading AF...", flush=True)
    AF = read_AF(AF_filepath, N)

    print("Diagonalizing by chunks...", flush=True)
    diagonalizing_chunked(hessian_filepath, AF, chunk_dir, chunk=CHUNK, save_vectors=SAVE_VECTORS)

    print("Assembling all chunks...", flush=True)
    eigenvalues, gamma = assemble(chunk_dir)
    assert eigenvalues.shape[0] == dim, "Missing blocks!"

    np.save(eigenvalues_filepath, eigenvalues)
    np.savetxt(gamma_data_filepath, gamma)
    np.savetxt(gamma_eigen_filepath, np.column_stack((eigenvalues, gamma)), header="Eigenvalue Gamma", comments="")

    print(f"T = {T} completed", flush=True)
    print("Data saved successfully")