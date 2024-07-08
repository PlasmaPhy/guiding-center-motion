"""This module initializes the "Particle" class, which calculates the orbit and can draw several different plots"""

import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.integrate import odeint


class Particle:
    def __init__(self, species, init_cond, mu, tspan, g=1):
        self.species = species
        self.init_cond = init_cond
        self.mu = mu
        self.g = g
        self.tspan = tspan

        # Logic variables
        self.calculated_orbit = False
        self.calculated_canonical = False

        # Grab configuration
        with open("config.json") as jsonfile:
            self.config = json.load(jsonfile)

    def __str__(self):
        return (
            f'Particle of Species:\t "{self.species}"\n'
            + f"Calculated orbit:\t {self.calculated_orbit}\n"
            + f"Calculated canonical:\t {self.calculated_canonical}\n"
        )

    def orbit(self):
        """Calculates the orbit of the particle and returns a numpy array."""
        
        def dSdt(t, S, mu=None):
            """Sets the diff equations system to pass to scipy"""

            theta, psi, z, Pz, rho = S
            # Intermediate values
            q = 1
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            r = np.sqrt(2 * psi)
            B = 1 - r * cos_theta
            par = self.mu + rho**2 * B

            theta_dot = 1 / q * (rho * B**2 - par * cos_theta / r)
            psi_dot = -1 / q * par * r * sin_theta
            z_dot = rho * B**2
            pz_dot = 0
            rho_dot = psi_dot

            return np.array([theta_dot, psi_dot, z_dot, pz_dot, rho_dot])

        self.sol = odeint(dSdt, y0=self.init_cond, t=self.tspan, tfirst=True)
        self.theta = self.sol.T[0]
        self.psi = self.sol.T[1]
        self.z = self.sol.T[2]
        self.Pz = self.sol.T[3]
        self.rho = self.sol.T[4]

        self.calculated_orbit = True
        self.canonical() # why not?

        return None

    def canonical(self):
        """Calculates the canonical variables"""

        self.Pz_can = self.rho * self.g - self.psi
        self.sol_can = self.sol.T
        self.sol_can[3] = self.Pz_can
        self.sol_can = self.sol_can.T

        self.calculated_canonical = True

        return None

    def plot_time_evolution(self, plot_canonical=True):
        """Plots the time evolution of the dynamical variabls"""

        if plot_canonical:
            theta, psi, z, Pz, rho = self.sol_can.T
        else:
            theta, psi, z, Pz, rho = self.sol.T
        
        # Set plot configurations from json file
        scatter_kw = {
            "s": self.config["time_plots_size"],
            "color": self.config["time_plots_color"],
        }
        ylabel_kw = {
            "rotation": 0,
            "fontsize": self.config["time_plots_ylabel_fontsize"],
        }

        # Plotting
        fig, ax = plt.subplots(5, 1, sharex=True)

        ax[0].scatter(self.tspan, theta, **scatter_kw)
        ax[1].scatter(self.tspan, z, **scatter_kw)
        ax[2].scatter(self.tspan, psi, **scatter_kw)
        ax[3].scatter(self.tspan, Pz, **scatter_kw)
        ax[4].scatter(self.tspan, rho, **scatter_kw)

        ax[0].set_ylabel("$\\theta(t)$\t", **ylabel_kw)
        ax[1].set_ylabel("$\\zeta(t)$\t", **ylabel_kw)
        ax[2].set_ylabel("$\\psi(t)$\t", **ylabel_kw)
        ax[3].set_ylabel("$P_\\zeta(t)$\t", **ylabel_kw)
        ax[4].set_ylabel("$\\rho(t)$\t", **ylabel_kw)

        plt.xlabel("$t$")
        
        return None

    def plot_drift(self, mod = False, plot_canonical=True):
        """Draws 2 plots: 1] θ-P_θ and 2] ζ-P_ζ
        """

        if plot_canonical:
            theta, psi, z, Pz, rho = self.sol_can.T
        else:
            theta, psi, z, Pz, rho = self.sol.T

        # Set plot configurations from json file
        scatter_kw = {
            "s": self.config["drift_plots_size"],
            "color": self.config["drift_plots_color"],
        }
        ylabel_kw = {
            "rotation": 0,
            "fontsize": self.config["drift_plots_ylabel_fontsize"],
        }
        xlabel_kw = {
            "rotation": 0,
            "fontsize": self.config["drift_plots_xlabel_fontsize"],
        }
        
        # Mods the x-axis (theta) to stay between -2π and +2π
        if mod: 
            theta = np.mod(theta, 2*np.pi)

        fig, ax = plt.subplots(1,2, figsize = (12,5))
        ax[0].scatter(theta, psi, **scatter_kw)
        ax[1].scatter(z, Pz, **scatter_kw)

        ax[0].set_xlabel("$\\theta$", **xlabel_kw)
        ax[1].set_xlabel("$\\zeta$", **xlabel_kw)

        ax[0].set_ylabel("$P_\\theta$", **ylabel_kw)
        ax[1].set_ylabel("$P_ζ$", **ylabel_kw)

        plt.sca(ax[0])
        plt.xticks(np.linspace(-np.pi, np.pi, 5), ["-π", "-π/2", "0", "π/2", "π"])

        return None