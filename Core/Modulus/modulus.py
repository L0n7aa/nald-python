import numpy as np
import os
import matplotlib.pyplot as plt
import config

# Parameters
N_atoms = config.N_atoms
N_molecules = config.N_molecules
T = config.T
id = config.id
dim = config.get_dim()

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
dir_name = os.path.join(script_dir + f'/../../Data/run_{id}_{T}_{N_atoms}_{N_molecules}')

eigen_gamma_path = os.path.join(dir_name, "eigen_gamma.npy")
eigenvalues_path = os.path.join(dir_name, "eigenvalues.npy")
gamma_path = os.path.join(dir_name, "gamma.npy")
volumes_path = os.path.join(dir_name, "volume.dat")
Gp_path = os.path.join(dir_name, "Gp.npy")
Gdp_path = os.path.join(dir_name, "Gdp.npy")

def load_data():
    volume = np.atleast_1d(np.loadtxt(volumes_path))[0]
    data = np.load(eigen_gamma_path)
    swp = data[:, 0]
    gamma0 = data[:, 1]
    return volume, swp, gamma0

def modif_gamma(swp, gamma, threshold):
    gamma = gamma.copy()
    epsilon = np.abs(swp) < 10**(-5)
    low = (swp > -threshold) & (swp < threshold) & ~epsilon
    gamma[epsilon] = 0.0
    gamma[low] = 65*np.abs(swp[low])
    return gamma


def compute():
    volume, swp, gamma0 = load_data()
    gamma_p = modif_gamma(swp, gamma0, threshold=1.01) # depend of the simulation
    s, g = swp[:-1], gamma0[:-1]
    gamma_dp = modif_gamma(s, g, threshold=1.0)
    w_all = np.exp(-15+ 0.1*np.arange(1,216)) # pourquoi 216 ?

    G_p = np.empty_like(w_all)
    G_dp = np.empty_like(w_all)

    for k, w in enumerate(w_all):
        d = swp - w**2
        G_p[k] = np.sum(-d * gamma_p / (d**2 + w**2)) / volume
        d = s - w**2
        G_dp[k] = np.sum(w * gamma_dp / (d**2 + w**2)) / volume

    np.save(Gp_path, np.column_stack([w_all, G_p]))
    np.save(Gdp_path, np.column_stack([w_all, G_dp]))