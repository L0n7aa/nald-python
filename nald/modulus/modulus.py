import numpy as np


def load_data(paths):
    volume = np.atleast_1d(np.loadtxt(paths.volume))[0]
    data = np.load(paths.eigen_gamma)
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


def compute(paths):
    volume, swp, gamma0 = load_data(paths)
    gamma_p = modif_gamma(swp, gamma0, threshold=1.01) 
    s, g = swp[:-1], gamma0[:-1]
    gamma_dp = modif_gamma(s, g, threshold=1.0)
    w_all = np.exp(-15+ 0.1*np.arange(1,216)) 

    G_p = np.empty_like(w_all)
    G_dp = np.empty_like(w_all)

    for k, w in enumerate(w_all):
        d = swp - w**2
        G_p[k] = np.sum(-d * gamma_p / (d**2 + w**2)) / volume
        d = s - w**2
        G_dp[k] = np.sum(w * gamma_dp / (d**2 + w**2)) / volume

    np.save(paths.gp, np.column_stack([w_all, G_p]))
    np.save(paths.gdp, np.column_stack([w_all, G_dp]))