"""This module initializes the "Particle" class, which calculates the orbit, orbit type, and can draw several different plots"""

import numpy as np
import matplotlib.pyplot as plt
import json
import Parabolas
import importlib
from scipy.integrate import odeint
importlib.reload(Parabolas) # for debugging

class Particle:
    """Constructs an individuial particle and sets its identity parameters,
    and offers plotting methods. Can also identify the type of orbit from
    the initial conditions.
    """

    def __init__(self, species, init_cond, mu, tspan, g=1, psi_wall=1):
        self.species = species
        self.init_cond = init_cond
        self.mu = mu
        self.g = g
        self.psi_wall = psi_wall
        self.tspan = tspan

        # Logic variables
        self.calculated_orbit = False
        self.calculated_canonical = False

        # Grab configuration
        self.grab_config()

    def __str__(self):
        string = f'Particle of Species:\t "{self.species}"\n' + \
                f"Calculated orbit:\t {self.calculated_orbit}\n\n"
        string += self.orbit_type(info=True)
        return string

    def grab_config(self):

        with open("config.json") as jsonfile:
            self.config = json.load(jsonfile)

        # Set plot configurations from json file

        self.time_scatter_kw = {
            "s": self.config["time_plots_size"],
            "color": self.config["time_plots_color"],
        }
        self.time_ylabel_kw = {
            "rotation": 0,
            "fontsize": self.config["time_plots_ylabel_fontsize"],
        }

        self.drift_scatter_kw = {
            "s": self.config["drift_scatter_size"],
            "color": self.config["drift_scatter_color"],
        }
        self.drift_plot_kw = {
            "linewidth": self.config["drift_plots_width"],
            "color": self.config["drift_plots_color"],
        }
        self.drift_ylabel_kw = {
            "rotation": 0,
            "fontsize": self.config["drift_plots_ylabel_fontsize"],
        }
        self.drift_xlabel_kw = {
            "rotation": 0,
            "fontsize": self.config["drift_plots_xlabel_fontsize"],
        }

    def orbit(self):
        """
        Calculates the orbit of the particle and returns a 2D numpy array,
        containing the 5 time evolution vectors of each dynamical variable.
        Orbit is stored in "self"
        """

        # Check if orbit has been already calculated
        if not self.calculated_orbit: 
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
            self.canonical()  # why not?

        return None

    def canonical(self):
        """Calculates the canonical variables and stores them in "self"."""

        # Only Pz changes
        self.Pz_can = self.rho * self.g - self.psi
        self.sol_can = self.sol.T
        self.sol_can[3] = self.Pz_can
        self.sol_can = self.sol_can.T

        self.calculated_canonical = True

        return None

    def orbit_type(self, info=True):
        """
        Calculates the orbit type given the initial conditions AND calculated obrit
        
        Trapped/passing:
        Te particle is trapped if rho vanishes, so we can 
        check if rho changes sign. Since rho = (2W - 2μB)^(1/2)/B, we need only to 
        check under the root.
        
        Confined/lost: 
        (from shape page 87 i guess)
        We only have to check if the particle is in-between the 2 left parabolas.

        """

        # Constants of Motion: Particle energy and Pz
        self.theta0, self.psi0, self.z0, self.Pz0, self.rho0 = self.init_cond
        self.r0 = np.sqrt(2 * self.psi0)
        self.B_init = 1 - self.r0 * np.cos(self.theta0)

        self.E = self.rho0**2 * self.B_init**2 / 2 + self.mu*self.B_init
        
        # Will be passed at __str__() to be printed in the end
        self.orbit_type_str =  "Constants of motion:\n" + \
                              f"Particle Energy:\tE = {self.E}\n" + \
                              f"Toroidal Momenta:\tPζ = {self.Pz0}\n"

        # Calculate Bmin and Bmax. In LAR, B decreases outwards.
        Bmin = 1 - np.sqrt(2*self.psi_wall)# "Bmin occurs at psi_wall, θ = 0"
        Bmax = 1 + np.sqrt(2*self.psi_wall) #Bmax occures at psi_wall, θ = π
     
        # Find if trapped or passing from rho (White page 83)
        if (2*self.E - 2*self.mu*Bmin)*(2*self.E - 2*self.mu*Bmax) < 0:
            self.t_or_p = "Trapped"
        else:
            self.t_or_p = "Passing"

        # Find if lost or confined 
        particle_x = self.Pz0/self.psi_wall
        particle_y = self.mu/self.E
        foo = Parabolas.Orbit_parabolas(self.E, psi_wall=self.psi_wall)
        #par2 = Parabolas.Parabola(self.abcs[1])

        upper_y = (foo.abcs[0][0]*particle_x**2 + foo.abcs[0][1]*particle_x +foo.abcs[0][2])
        lower_y = (foo.abcs[1][0]*particle_x**2 + foo.abcs[1][1]*particle_x +foo.abcs[1][2])

        if particle_y < upper_y and particle_y > lower_y:
            self.l_or_c = "Lost"
        else:
            self.l_or_c = "Confined"

        self.orbit_type_str =  "Constants of motion:\n" + \
                      f"Particle Energy:\tE = {np.around(self.E,8)}\n" + \
                      f"Toroidal Momenta:\tPζ = {self.Pz0}\n\n" + \
                      f"Orbit Type: {self.t_or_p} - {self.l_or_c}\n"

        if info:
            return self.orbit_type_str

    def calcW_grid(self, theta, psi, Pz):
        """Returns a single value or a grid of the calculated Hamiltonian."""
        r = np.sqrt(2 * psi)
        B = 1 - r * np.cos(theta)
        # psi_p = psi
        return (Pz + psi) ** 2 * B**2 / (2 * self.g**2) + self.mu * B

    def plot_time_evolution(self, plot_canonical=True):
        """Plots the time evolution of the dynamical variabls"""

        if plot_canonical:
            theta, psi, z, Pz, rho = self.sol_can.T
        else:
            theta, psi, z, Pz, rho = self.sol.T

        # Plotting
        fig, ax = plt.subplots(5, 1, sharex=True)
        fig.suptitle("Time evolution of dynamical variables")

        ax[0].scatter(self.tspan, theta, **self.time_scatter_kw)
        ax[1].scatter(self.tspan, z, **self.time_scatter_kw)
        ax[2].scatter(self.tspan, psi, **self.time_scatter_kw)
        ax[3].scatter(self.tspan, Pz, **self.time_scatter_kw)
        ax[4].scatter(self.tspan, rho, **self.time_scatter_kw)

        ax[0].set_ylabel("$\\theta(t)$\t", **self.time_ylabel_kw)
        ax[1].set_ylabel("$\\zeta(t)$\t", **self.time_ylabel_kw)
        ax[2].set_ylabel("$\\psi(t)$\t", **self.time_ylabel_kw)
        ax[3].set_ylabel("$P_\\zeta(t)$\t", **self.time_ylabel_kw)
        ax[4].set_ylabel("$\\rho(t)$\t", **self.time_ylabel_kw)

        plt.xlabel("$t$")

        return None

    def plot_drift(self, theta_lim, plot_canonical=True):
        """Draws 2 plots: 1] θ-P_θ and 2] ζ-P_ζ"""

        if plot_canonical:
            theta, psi, z, Pz, rho = self.sol_can.T
        else:
            theta, psi, z, Pz, rho = self.sol.T

        if theta_lim == [0,2*np.pi]:
            self.theta_plot = np.mod(self.theta, 2*np.pi)
            self.theta_min, self.theta_max = theta_lim
        elif theta_lim == [-np.pi, np.pi]:
            theta_plot = np.mod(self.theta, 2*np.pi)
            self.theta_plot = theta_plot - 2*np.pi*(theta_plot > np.pi)
            self.theta_min, self.theta_max = theta_lim
        else: 
            print("theta_lim must be either [0,2*np.pi] or [-np.pi,np.pi].")
            return

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Drift orbits of $P_\\theta - \\theta$ and $P_\zeta - \zeta$")

        ax[0].scatter(self.theta_plot, psi, **self.drift_scatter_kw)
        ax[1].plot(z, Pz, **self.drift_plot_kw)

        ax[0].set_xlabel("$\\theta$", **self.drift_xlabel_kw)
        ax[1].set_xlabel("$\\zeta$", **self.drift_xlabel_kw)

        ax[0].set_ylabel("$P_\\theta$", **self.drift_ylabel_kw)
        ax[1].set_ylabel("$P_ζ$", **self.drift_ylabel_kw)

        plt.sca(ax[0])
        # Store plot lims for contour plot
        self.drift_xlim = ax[0].get_xlim()
        self.drift_ylim = ax[0].get_ylim()

        # Set all xticks as multiples of π, and then re-set xlims (smart!)
        ticks = ["-2π", "-3π/2", "-π", "-π/2", "0", "π/2", "π", "3π/2", "2π"]
        plt.xticks(np.linspace(-2*np.pi, 2*np.pi, 9), ticks)
        plt.xlim(theta_lim)

        return None

    def contour_energy(self, scatter = True, theta_lim = "auto", psi_lim = "auto", Pz = None, levels=None):
        """Draws a 2D contour plot of the Hamiltonian"""
        
        # Set grid limits

        if psi_lim == "auto": # Use ylim from drift plots
            psi_min, psi_max = self.drift_ylim
        else:
            psi_min, psi_max = psi_lim

        # Calculate Energy values
        grid_density = self.config["contour_grid_density"]
        theta, psi = np.meshgrid(np.linspace(self.theta_min, self.theta_max, grid_density),
                                 np.linspace(psi_min, psi_max, grid_density))
        values = self.calcW_grid(theta, psi, Pz)
        span = np.array([values.min(), values.max()])

        # Create Figure
        if levels is None: # If non is given
            levels = self.config["contour_levels_default"]
        contour_kw = {
            "vmin": span[0], "vmax": span[1], 
            "levels": levels, "cmap": self.config["contour_cmap"]
        }
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        ax.set_facecolor('white')

        # Contour plot
        C = ax.contourf(theta, psi, values, **contour_kw)
        ax.set_xlabel("$\\theta$")
        ax.set_ylabel("$\\psi_p\t$", rotation = 0)
        plt.scatter(self.theta_plot, self.psi, **self.drift_scatter_kw)
        ticks = ["-2π", "-3π/2", "-π", "-π/2", "0", "π/2", "π", "3π/2", "2π"]
        plt.xticks(np.linspace(-2*np.pi, 2*np.pi, 9), ticks)        
        ax.set(xlim = [self.theta_min, self.theta_max], ylim = [psi_min, psi_max])
        fig.colorbar(C, ax=ax, fraction=0.03, pad=0.2)        

        return None

    def plot_orbit_type_point(self):
        B0 = 1
        x = self.Pz0/self.psi_wall
        y = self.mu*B0/self.E

        orbit_point_kw = {
            "markersize": self.config["orbit_point_size"],
            "marker": self.config["orbit_point_marker"],
            "markeredgecolor": self.config["orbit_point_edge_color"],
            "markerfacecolor": self.config["orbit_point_face_color"]
        }

        plt.plot(x, y, **orbit_point_kw)
        label = "  Particle " + f"({self.t_or_p[0]}-{self.l_or_c[0]})"
        plt.annotate(label, (x, y))
        return