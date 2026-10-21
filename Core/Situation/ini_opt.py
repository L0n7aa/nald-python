import numpy as np
import os
import config

# ---------- Parameters ----------
N_atoms = config.N_atoms
N_molecules = config.N_molecules
T = config.T
id = config.id

L = np.array([30, 30, 30])  # Dimensions of the cube            # Atoms per molecule
step = 1.0                  # Length of the walk
D_MIN = 0.9                 # Minimum distance between two atoms
max_tries_per_step = 50     # Max tries per step before restarting
rng = np.random.default_rng()
sigma = 0.5

def paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    id = len(os.listdir(os.path.join(script_dir, '../../Data')))
    dir_name = os.path.join(script_dir + f'/../../Data/run_{id}_{T}_{N_atoms}_{N_molecules}')
    os.makedirs(dir_name)
    npy_data_path = os.path.join(dir_name, 'molecules_opt.npy')
    atom_path = os.path.join(dir_name, 'atoms.csv')
    image_path = os.path.join(dir_name, 'visualization.png')
    return npy_data_path, atom_path, image_path, id

# Optimization: Spatial Grid to avoid O(N) loops
cell_size = D_MIN 
grid_shape = np.ceil(L / cell_size).astype(int)
grid_cache = {}  # Stores atoms by cell

def get_grid_key(p):
    return tuple((p / cell_size).astype(int))

def init_grid():
    global grid_cache
    grid_shape = tuple(np.ceil(L / cell_size).astype(int))

def add_to_grid(p):
    key = get_grid_key(p)
    if key in grid_cache:
        grid_cache[key].append(p)

def check_neighbors(p):
    """Checks if p is valid by looking only at neighboring cells."""
    key = get_grid_key(p)
    x, y, z = key
    
    # Check the cell itself and the 26 neighbors
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < grid_shape[0] and 0 <= ny < grid_shape[1] and 0 <= nz < grid_shape[2]:
                    neighbor_key = (nx, ny, nz)
                    if neighbor_key in grid_cache:
                        for neighbor in grid_cache[neighbor_key]:
                            dist_sq = np.sum((neighbor - p) ** 2)
                            if dist_sq < D_MIN ** 2:
                                return False
    return True

def uniform_direction():
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)

def disturbed_direction(d_prec, sigma):
    v = d_prec + sigma * uniform_direction()
    norm_v = np.linalg.norm(v)
    return v / norm_v if norm_v > 1e-8 else uniform_direction()

def is_inside_cube(p):
    """Check if point p is strictly inside the cube [0, L)."""
    # Use .all() to ensure all coordinates satisfy the condition
    # We explicitly cast to bool to avoid ambiguity
    return bool(np.all(p >= 0) and np.all(p < L))

def generate_molecule_optimized():
    """Generates a molecule using the spatial grid."""
    
    # Attempt initial placement
    max_start_attempts = 1000
    p0 = None
    for _ in range(max_start_attempts):
        candidate_p0 = rng.uniform(0, L)
        if check_neighbors(candidate_p0) and is_inside_cube(candidate_p0):
            p0 = candidate_p0
            break
            
    if p0 is None:
        return None

    molecule = np.empty((N_atoms, 3))
    molecule[0] = p0
    add_to_grid(p0)
    
    current_dir = uniform_direction()
    
    for i in range(1, N_atoms):
        success_step = False
        tries = 0
        
        while not success_step and tries < max_tries_per_step:
            candidate = molecule[i-1] + step * disturbed_direction(current_dir, sigma)
            
            # Check cube boundaries explicitly
            if not is_inside_cube(candidate):
                tries += 1
                continue
                
            # Distance check via grid (much faster)
            if check_neighbors(candidate):
                molecule[i] = candidate
                add_to_grid(candidate)
                current_dir = disturbed_direction(current_dir, sigma) # Update direction
                success_step = True
            else:
                tries += 1
        
        if not success_step:
            # Failure: we cancel everything and return None
            return None
            
    return molecule

def generate_system_optimized():
    all_atoms_count = 0
    molecules_list = [None] * N_molecules
    
    # Initialize empty grid
    init_grid()
    
    attempts_total = 0
    max_attempts_global = N_molecules * 10000 # Safety against infinite total loop
    
    idx = 0
    while idx < N_molecules:
        mol = generate_molecule_optimized()
        
        if mol is not None:
            molecules_list[idx] = mol
            all_atoms_count += N_atoms
            idx += 1
        else:
            pass
            
        attempts_total += 1
        if attempts_total > max_attempts_global:
            print(f"Stopping after {attempts_total} total attempts. The system is too dense.")
            break
            
    # Filter out any None values if present
    valid_mols = [m for m in molecules_list if m is not None]
    return np.array(valid_mols)