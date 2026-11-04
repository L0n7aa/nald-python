# Simulation Analysis Tool (`nald.py`)

## Prerequisites

Ensure you have Python 3.8+ installed along with the required libraries.

### Core Dependencies
- **NumPy**
- **SciPy**
- **Matplotlib**
- **LAMMPS** 

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/L0n7aa/nald-python.git
   cd nald-python
   ```

## Installing LAMMPS (with MPI and Python bindings)

`nald.py` runs LAMMPS through its Python module and uses MPI to run on several cores.

1. an MPI implementation (OpenMPI),
2. LAMMPS compiled **with MPI** and as a **shared library**,
3. the `lammps` Python module,
4. `mpi4py`, built against the **same** MPI.

### Option A: conda (recommended, no compilation)

```bash
conda create -n nald -c conda-forge python lammps mpi4py openmpi numpy scipy matplotlib
conda activate nald
```

Then go to [Checking the installation](#checking-the-installation).

### Option B: build from source (Fedora; use `apt` equivalents on Debian/Ubuntu)

**1. System packages**

**On Debian/Ubuntu** : 

```bash
sudo apt install build-essential cmake python3-dev openmpi-bin libompi-dev
which mpirun mpicc mpicxx
```

**On Fedora/RHEL** :

```bash
sudo dnf install gcc-c++ cmake make python3-devel openmpi openmpi-devel
module load mpi/openmpi-x86_64     # repeat in every new terminal
which mpirun mpicc mpicxx          # all three must be found
```

**2. Python environment**

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
MPICC=$(which mpicc) pip install --no-binary=mpi4py mpi4py
```

`--no-binary` forces `mpi4py` to be compiled against your own MPI.

**3. Build LAMMPS** (from the LAMMPS source directory, in a fresh `build` folder)

```bash
mkdir build && cd build
cmake ../cmake \
      -D BUILD_MPI=yes \
      -D BUILD_SHARED_LIBS=yes \
      -D PKG_MOLECULE=yes \
      -D CMAKE_INSTALL_PREFIX=$VIRTUAL_ENV \
      -D CMAKE_INSTALL_RPATH_USE_LINK_PATH=yes
cmake --build . -j 8
```

Add one `-D PKG_<NAME>=yes` for each package required by the `in.*` scripts.

**4. Install LAMMPS and its Python module**

```bash
cmake --install .
cmake --build . --target install-python
```

### Checking the installation

```bash
lmp -h | grep "MPI v"          # must NOT say "STUBS"
python -c "from lammps import lammps; print(lammps().version())"
mpirun -np 4 python -c "from mpi4py import MPI; c = MPI.COMM_WORLD; print(c.Get_rank(), c.Get_size())"
```

The last command must print four different ranks (`0 4`, `1 4`, `2 4`, `3 4`).
If it prints `0 1` four times, `mpi4py` is not linked to the same MPI as `mpirun`.

---

## Usage

Run the main orchestrator script `nald.py` by providing the system parameters via command-line options.

### Command Syntax

```bash
python3 nald.py -M <molecules> -N <atoms> -T <temperature> -n <MPI processes> -r <simulation> -j <simulations at the same time> --lammps <OPTIONNAL : path to the LAMMPS executable> --yes <OPTIONNAL : non-interactive mode>
```

### Parameters

| Argument | Short | Description | Example |
| :--- | :--- | :--- | :--- |
| `--molecules` | `-M` | Number of molecules in the system | `50` |
| `--atoms` | `-A` | Number of atoms per molecule | `50` |
| `--temperature` | `-T` | System simulation temperature | `0.5` |
| `--np` | `-n` | MPI processes per simulation | `4` |
| `--runs` | `-r` | Number of simulations | `2` |
| `--jobs` | `-j` | Number of simulations running at the same time | `2` |
| `--lammps` | / | **OPTIONNAL** : Path to the LAMMPS executable` | `path/to/LAMMPS` |
| `--yes` | / | **OPTIONNAL** : Non intervative mode` | / | 

### Example

```bash
python3 nald.py -M 50 -A 50 -T 0.5 -n 2 -r 4 -j 2 --yes
```

---

## Project Structure

```text
.
├── nald.py                    # Main entry point (orchestrator)
├── config.py                  # Global configuration and parameter management
├── requirements.txt           # Dependency declarations
├── README.md                  # Project documentation
│
├── Structure/                 # Initial setup and system geometry
│   ├── __init__.py
│   ├── ini_opt.py             # System generation functions
│   ├── show_data.py           # Functions for saving the data to .csv files and plotting it
│   └── structure.py           # Main function of the module
│
├── LAMMPS/                    # LAMMPS script execution
│   ├── __init__.py
│   ├── convert_data_bond.py   # Converts the .csv files into a LAMMPS input (data) file
│   ├── lammps_runner.py       # Functions for running LAMMPS with the chosen parameters
│   └── lammps_scripts.py      # Main function of the module
│
├── Diagonalization/           # Matrix and spectral analysis
│   ├── __init__.py
│   ├── diagonalization.py     # Functions for symmetrizing and diagonalizing the matrix
│   ├── hessian.py             # Functions for reconstructing the Hessian from the LAMMPS output files
│   └── scripts_diag.py        # Main function of the module
│
└── Modulus/                   # Modulus computation tools
    ├── __init__.py
    └── modulus.py             # Main function of the module: computes G′ and G″ from the diagonalization output files
```
---

## Data Storage

Each simulation result can be found in the `Data/run_{id}_{temperature}_{number of atoms}_{number of molecules}` directory.

## Logs Storage

LAMMPS logs can be found in the `Core/Logs/run{id}` directory.