# Simulation Analysis Tool (`nald.py`)

## Prerequisites

Ensure you have Python 3.8+ installed along with the required libraries.

### Core Dependencies
- **NumPy**
- **SciPy**
- **Matplotlib**
- **LAMMPS** (with Python bindings)

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/L0n7aa/nald-python.git
   cd nald-python
   ```

2. **Create your data storage directory:**
   ```bash
   mkdir Data
   ```

3. **(Optional) Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies:**
   ```bash
   cd Core
   pip install -r requirements.txt
   ```

> **Note on LAMMPS:** Ensure the C++ LAMMPS shared library (`liblammps.so` or `liblammps.dylib`) is compiled and accessible in your environment path or installed via Conda (`conda install -c conda-forge lammps`).

---

## Usage

Run the main orchestrator script `nald.py` by providing the system parameters via command-line options.

### Command Syntax

```bash
python3 nald.py -M <molecules> -N <atoms> -T <temperature>
```

### Parameters

| Argument | Short | Description | Example |
| :--- | :--- | :--- | :--- |
| `--molecules` | `-M` | Number of molecules in the system | `50` |
| `--atoms` | `-A` | Number of atoms per molecule | `50` |
| `--temperature` | `-T` | System simulation temperature | `0.5` |

### Example

```bash
python3 nald.py -M 50 -N 50 -T 0.5
```

During execution, the script may prompt you interactively to regenerate or adjust the generated system before proceeding with downstream calculations.

---

## Project Structure

```text
.
├── nald.py                  # Main entry point (orchestrator)
├── config.py                # Global configuration & parameter management
├── requirements.txt         # Dependency declarations
├── README.md                # Project documentation
│
├── Situation/               # Module for initial setup and system geometry
│   ├── __init__.py
│   └── situation.py
│
├── LAMMPS/                  # Module for LAMMPS script execution
│   ├── __init__.py
│   └── lammps_scripts.py
│
├── Diagonalization/         # Matrix and spectral analysis
│   ├── __init__.py
│   └── scripts_diag.py
│
└── Modulus/                 # Modulus computation tools
    ├── __init__.py
    └── modulus.py
```
---

## Data Storage

Each simulation result can be found in the `Data/run_{id}_{temperature}_{number of atoms}_{number of molecules}` directory.