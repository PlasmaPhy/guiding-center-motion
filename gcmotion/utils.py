import sys
import yaml
import importlib
import numpy as np
from pathlib import Path


def theta_plot(theta, theta_lim=[0, 2 * np.pi]):
    if theta_lim == [0, 2 * np.pi]:
        theta_plot = np.mod(theta, 2 * np.pi)
    elif theta_lim == [-np.pi, np.pi]:
        theta_plot = np.mod(theta, 2 * np.pi)
        theta_plot = theta_plot - 2 * np.pi * (theta_plot > np.pi)
    else:
        print("theta_lim must be either [0,2*np.pi] or [-np.pi,np.pi].")
        print("Defaulting to [-π, π].")
        theta_plot = np.mod(theta, 2 * np.pi)
        theta_plot = theta_plot - 2 * np.pi * (theta_plot > np.pi)
        return
    return theta_plot


def reload():
    for mod in [mod for name, mod in sys.modules.items() if "gcmotion" in name]:
        importlib.reload(mod)
