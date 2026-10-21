import numpy as np
import os


def organize(dim, old_hessian, new_hessian):
    """Read the Hessian from file and save it incrementally via memory-mapping."""
    # Create a memory-mapped file: only the current block is in RAM
    hessian = np.memmap(new_hessian, dtype=np.float64, mode="w+", shape=(dim, dim))

    with open(old_hessian, "r") as f:
        for row in range(dim):
            for col in range(0, dim, 3):
                line = f.readline()
                hessian[row, col:col + 3] = np.array(line.split(), dtype=np.float64)
            # Flush every 1000 rows to keep disk I/O steady
            if (row + 1) % 1000 == 0:
                hessian.flush()
                print(f"Row {row + 1}/{dim} written", flush=True)

    hessian.flush()
    del hessian
    print(f"Hessian saved to {new_hessian}", flush=True)

def symetrizing(dim, filepath):
    """Load the Hessian, symetrize it and save it in hessian.py."""
    print("Started symetrizing hessian")
    hessian = np.fromfile(filepath, dtype=np.float64, count=dim*dim).reshape(dim, dim)
    hessian += hessian.T
    hessian *= 0.5
    np.save(filepath, hessian)
    print("Symetrizing done")