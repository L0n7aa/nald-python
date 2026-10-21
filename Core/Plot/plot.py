import matplotlib.pyplot as plt
import numpy as np
import os
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

Gp_path = os.path.join(dir_name, "Gp.npy")
Gdp_path = os.path.join(dir_name, "Gdp.npy")

def plot_moduli():
  gp_data = np.load(Gp_path)
  gdp_data = np.load(Gdp_path)

  w_all = gp_data[:, 0]
  G_p = gp_data[:, 1]  
  G_dp = gdp_data[:, 1]  

  fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

  axes[0].plot(w_all, G_p, label=f'T = {T}', color='black')
  axes[0].set_ylabel(r"$G'$", fontsize=12)
  axes[0].set_xscale('log')
  axes[0].grid(True, which='both', linestyle='--', alpha=0.5)
  axes[0].legend()

  axes[1].plot(w_all, G_dp, label=f'T = {T}', color='black')
  axes[1].set_xlabel(r'$\Omega$', fontsize=12)
  axes[1].set_ylabel(r"$G''$", fontsize=12)
  axes[1].set_xscale('log')
  axes[1].set_yscale('log')  
  axes[1].grid(True, which='both', linestyle='--', alpha=0.5)
  axes[1].legend()

  plt.tight_layout()
  plt.show()