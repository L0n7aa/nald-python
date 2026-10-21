N_atoms = None
N_molecules = None
id = None
T = None

def get_dim():
    if N_atoms is None or N_molecules is None:
        return None
    return 3*N_atoms*N_molecules