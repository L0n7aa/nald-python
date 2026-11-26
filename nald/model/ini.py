import numpy as np


class SystemGenerator:
    STEP = 1.0                 
    D_MIN = 0.9                
    MAX_TRIES_PER_STEP = 50
    SIGMA = 0.5

    def __init__(self, n_atoms, n_molecules, density, seed=None):
        self.n_atoms = n_atoms
        self.n_molecules = n_molecules
        V = n_atoms * n_molecules / density
        length = V ** (1 / 3)
        self.L = np.array([length, length, length])
        self.rng = np.random.default_rng(seed)
        self.cell_size = self.D_MIN
        self.grid_shape = np.ceil(self.L / self.cell_size).astype(int)
        self.grid_cache = {}

    def get_grid_key(self, p):
        return tuple((p / self.cell_size).astype(int))

    def init_grid(self):
        self.grid_cache.clear()

    def add_to_grid(self, p):
        self.grid_cache.setdefault(self.get_grid_key(p), []).append(p)

    def remove_from_grid(self, points):
        for p in points:
            cell = self.grid_cache[self.get_grid_key(p)]
            cell[:] = [q for q in cell if q is not p]

    def check_neighbors(self, p):
        x, y, z = self.get_grid_key(p)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if (0 <= nx < self.grid_shape[0] and 0 <= ny < self.grid_shape[1]
                            and 0 <= nz < self.grid_shape[2]):
                        for neighbor in self.grid_cache.get((nx, ny, nz), ()):
                            if np.sum((neighbor - p) ** 2) < self.D_MIN ** 2:
                                return False
        return True

    def uniform_direction(self):
        v = self.rng.normal(size=3)
        return v / np.linalg.norm(v)

    def disturbed_direction(self, d_prec, sigma):
        v = d_prec + sigma * self.uniform_direction()
        norm_v = np.linalg.norm(v)
        return v / norm_v if norm_v > 1e-8 else self.uniform_direction()

    def is_inside_cube(self, p):
        return bool(np.all(p >= 0) and np.all(p < self.L))

    def generate_molecule(self):
        p0 = None
        for _ in range(1000):
            candidate_p0 = self.rng.uniform(0, self.L)
            if self.check_neighbors(candidate_p0) and self.is_inside_cube(candidate_p0):
                p0 = candidate_p0
                break
        if p0 is None:
            return None

        molecule = np.empty((self.n_atoms, 3))
        molecule[0] = p0
        self.add_to_grid(p0)
        placed = [p0]
        current_dir = self.uniform_direction()

        for i in range(1, self.n_atoms):
            success_step = False
            tries = 0
            while not success_step and tries < self.MAX_TRIES_PER_STEP:
                candidate = molecule[i - 1] + self.STEP * self.disturbed_direction(current_dir, self.SIGMA)
                if not self.is_inside_cube(candidate):
                    tries += 1
                    continue
                if self.check_neighbors(candidate):
                    molecule[i] = candidate
                    self.add_to_grid(candidate)
                    placed.append(candidate)
                    current_dir = self.disturbed_direction(current_dir, self.SIGMA)
                    success_step = True
                else:
                    tries += 1
            if not success_step:
                self.remove_from_grid(placed)
                return None
        return molecule

    def generate_system(self):
        molecules_list = [None] * self.n_molecules
        self.init_grid()
        attempts_total = 0
        max_attempts_global = self.n_molecules * 10000
        idx = 0
        while idx < self.n_molecules:
            mol = self.generate_molecule()
            if mol is not None:
                molecules_list[idx] = mol
                idx += 1
            attempts_total += 1
            if attempts_total > max_attempts_global:
                print(f"Stopping after {attempts_total} total attempts. The system is too dense.")
                break
        return np.array([m for m in molecules_list if m is not None])