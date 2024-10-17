import numpy as np


def pi_mod(x, x_lim=[0, 2 * np.pi]):
    if x_lim == [0, 2 * np.pi]:
        x_plot = np.mod(x, 2 * np.pi)
    elif x_lim == [-np.pi, np.pi]:
        x_plot = np.mod(x, 2 * np.pi)
        x_plot = x_plot - 2 * np.pi * (x_plot > np.pi)
    else:
        print("x_lim must be either [0,2*np.pi] or [-np.pi,np.pi].")
        print("Defaulting to [-π, π].")
        x_plot = np.mod(x, 2 * np.pi)
        x_plot = x_plot - 2 * np.pi * (x_plot > np.pi)
        return
    return x_plot
