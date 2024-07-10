"""
This module initializes the "Particle" class, which calculates the orbit,
orbit type, and can draw several different plots
"""

import numpy as np
import matplotlib.pyplot as plt
import utils
import Parabolas
from importlib import reload
from scipy.integrate import odeint

reload(Parabolas)  # for debugging


class Particle:
    """Initializes a particle.

    To change the canonical equations, go to self.dSdt(), under
    self.orbit().

    To change q factor, first change the expression in self.q(), then
    solve for psi and change the expression in self.psi_from_psip(),
    and self.psip_from_psi()"""

    def __init__(self, species, init_cond, mu, tspan, B=[0, 1, 0], psip_wall=0.4):
        """Initializes particle and grabs configuration.

        Args:
            species (str): the particle species
            init_cond (np.array): 1x5 initial conditions array
            mu (float): magnetic moment
            tspan (np.array): The ODE interval, in [t0, tf, steps]
            B (list, optional): The 3 componets of the contravariant representation
                                of the magnetic field B. Defaults to [0, 1, 0].
            psip_wall (float, optional): The value of ψ at the wall. Better be lower
                                lower than 0.5. Defaults to 0.4.
        """
        # Initialization
        self.species = species
        self.init_cond = init_cond
        self.mu = mu
        self.B = B
        self.I, self.g, self.delta = self.B
        self.psip_wall = psip_wall
        self.tspan = tspan

        # Logic variables
        self.calculated_orbit = False
        self.t_or_p = "Unknown"
        self.l_or_c = "Unknown"

        # Grab configuration
        self.Config = utils.Config_file()

    def __str__(self):
        string = (
            f'Particle of Species:\t "{self.species}"\n'
            + f"Calculated orbit:\t {self.calculated_orbit}\n\n"
        )
        # Also grab orbit_type() results
        string += self.orbit_type(info=True)
        return string

    def orbit(self):
        """Calculates the orbit of the particle.

        Calculates a 2D numpy array, containing the 4 time evolution vectors of
        each dynamical variable (θ, ψ_p, ζ, ρ). Afterwards, it calculates the
        canonical momenta (P_θ, P_ζ).
        Orbit is stored in "self"
        """

        # Check if orbit has been already calculated
        if self.calculated_orbit:
            return

        print("Calculating Orbit")

        def dSdt(t, S, mu=None):
            """Sets the diff equations system to pass to scipy"""

            theta, psip, z, rho = S

            # ψ is calculated in each step. This can be done since
            # we know ψ_p and q.
            psi = self.psi_from_psip(psip)

            # Intermediate values
            q = self.q(psi)
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            r = np.sqrt(2 * psi)
            B = 1 - r * cos_theta
            par = self.mu + rho**2 * B

            # Canonical Equation (Change this)
            theta_dot = 1 / q * (rho * B**2 - par * cos_theta / r)
            psip_dot = -1 / q * par * r * sin_theta
            z_dot = rho * B**2
            rho_dot = psip_dot

            return np.array([theta_dot, psip_dot, z_dot, rho_dot])

        sol_init = np.delete(self.init_cond, 3)  # Drop Pz0
        self.sol = odeint(dSdt, y0=sol_init, t=self.tspan, tfirst=True)

        self.theta = self.sol.T[0]
        self.psip = self.sol.T[1]
        self.z = self.sol.T[2]
        self.rho = self.sol.T[3]

        # ψ must be calculated again afterwards, since it is not calculated
        # from the ODE system, but the q factor
        self.psi = self.psi_from_psip(self.psip)

        self.calculated_orbit = True
        self.calc_momenta()

    def calc_momenta(self):
        """Calculates the canonical momenta P_θ and P_ζ and stores them in "self"."""

        self.Ptheta = self.psi + self.rho * self.I
        self.Pzeta = self.rho * self.g - self.psip

    def q(self, psi):
        """Returns q(ψ)."""
        return 1 + psi**2

    def psi_from_psip(self, psip):
        """Returns ψ(ψ_p), as defined from the safety factor q.

        Only applicable if q can be solved for ψ (q can be integrated
        with respect to ψ). For more complex form of q, another method
        is required.
        """
        return np.tan(psip)

    def psip_from_psi(self, psi):
        """Returns ψ_p(ψ), as defined from the safety factor q.

        Only applicable if q can be solved for ψ_p (q can be integrated
        with respect to ψ). For more complex form of q, another method
        is required.
        """
        return np.atan(psi)

    def orbit_type(self, info=True):
        """
        Calculates the orbit type given the initial conditions ONLY.

        Trapped/passing:
        Te particle is trapped if rho vanishes, so we can
        check if rho changes sign. Since rho = (2W - 2μB)^(1/2)/B, we need only to
        check under the root.

        Confined/lost:
        (from shape page 87 i guess)
        We only have to check if the particle is in-between the 2 left parabolas.

        """

        # Constants of Motion: Particle energy and Pz
        self.theta0, self.psip0, self.z0, self.Pz0, self.rho0 = self.init_cond
        self.r0 = np.sqrt(2 * self.psip0)
        self.B_init = 1 - self.r0 * np.cos(self.theta0)

        self.E = self.rho0**2 * self.B_init**2 / 2 + self.mu * self.B_init

        # Will be passed at __str__() to be printed in the end
        self.orbit_type_str = (
            "Constants of motion:\n"
            + f"Particle Energy:\tE = {self.E}\n"
            + f"Toroidal Momenta:\tPζ = {self.Pz0}\n"
        )

        # Calculate Bmin and Bmax. In LAR, B decreases outwards.
        Bmin = 1 - np.sqrt(2 * self.psip_wall)  # "Bmin occurs at psip_wall, θ = 0"
        Bmax = 1 + np.sqrt(2 * self.psip_wall)  # "Bmax occurs at psip_wall, θ = π"

        # Find if trapped or passing from rho (White page 83)
        if (2 * self.E - 2 * self.mu * Bmin) * (2 * self.E - 2 * self.mu * Bmax) < 0:
            self.t_or_p = "Trapped"
        else:
            self.t_or_p = "Passing"

        # Find if lost or confined
        particle_x = self.Pz0 / self.psip_wall
        particle_y = self.mu / self.E
        foo = Parabolas.Orbit_parabolas(self.E, psip_wall=self.psip_wall)

        # Recalculate y by reconstructing the parabola (there might be a better way
        # to do this)
        upper_y = (
            foo.abcs[0][0] * particle_x**2
            + foo.abcs[0][1] * particle_x
            + foo.abcs[0][2]
        )
        lower_y = (
            foo.abcs[1][0] * particle_x**2
            + foo.abcs[1][1] * particle_x
            + foo.abcs[1][2]
        )

        if particle_y < upper_y and particle_y > lower_y:
            self.l_or_c = "Lost"
        else:
            self.l_or_c = "Confined"

        self.orbit_type_str = (
            "Constants of motion:\n"
            + f"Particle Energy:\tE = {np.around(self.E,8)}\n"
            + f"Toroidal Momenta:\tPζ = {self.Pz0}\n\n"
            + f"Orbit Type: {self.t_or_p} - {self.l_or_c}\n"
        )

        if info:
            return self.orbit_type_str

    def calcW_grid(self, theta, psi, Pz):
        """Returns a single value or a grid of the calculated Hamiltonian."""

        r = np.sqrt(2 * psi)
        B = 1 - r * np.cos(theta)
        psip = self.psip_from_psi(psi)
        return (Pz + psip) ** 2 * B**2 / (2 * self.g**2) + self.mu * B

    def plot_time_evolution(self, percentage=100):
        """
        Plots the time evolution of the dynamical variabls and
        canonical momenta.

        Args:
        percentage (int): 0-100: the percentage of the orbit to be plotted
        """

        points = int(np.floor(self.theta.shape[0] * percentage / 100) - 1)

        # Plotting
        fig, ax = plt.subplots(6, 1, figsize=(10, 8), sharex=True)
        fig.tight_layout()
        ax[0].title.set_text("Time evolution of dynamical variables")
        ax[4].title.set_text("Time evolution of canonical momenta")

        ax[0].scatter(self.tspan[:points], self.theta[:points], **self.Config.time_scatter_kw)
        ax[1].scatter(self.tspan[:points], self.z[:points], **self.Config.time_scatter_kw)
        ax[2].scatter(self.tspan[:points], self.psi[:points], **self.Config.time_scatter_kw)
        ax[3].scatter(self.tspan[:points], self.rho[:points], **self.Config.time_scatter_kw)
        ax[4].scatter(self.tspan[:points], self.Ptheta[:points], **self.Config.time_scatter_kw)
        ax[5].scatter(self.tspan[:points], self.Pzeta[:points], **self.Config.time_scatter_kw)

        ax[0].set_ylabel("$\\theta(t)$\t", **self.Config.time_ylabel_kw)
        ax[1].set_ylabel("$\\zeta(t)$\t", **self.Config.time_ylabel_kw)
        ax[2].set_ylabel("$\\psi(t)$\t", **self.Config.time_ylabel_kw)
        ax[3].set_ylabel("$\\rho(t)$\t", **self.Config.time_ylabel_kw)
        ax[4].set_ylabel("$P_\\theta(t)$\t\t", **self.Config.time_ylabel_kw)
        ax[5].set_ylabel("$P_\\zeta(t)$\t", **self.Config.time_ylabel_kw)

        plt.xlabel("$t$")

    def plot_drift(self, theta_lim):
        """Draws 2 plots: 1] θ-P_θ and 2] ζ-P_ζ

        Args:
            theta_lim (np.array): x-axis limits
        """

        # Set theta lim. Mods all thetas to 2π
        if theta_lim == [0, 2 * np.pi]:
            self.theta_plot = np.mod(self.theta, 2 * np.pi)
            self.theta_min, self.theta_max = theta_lim
        elif theta_lim == [-np.pi, np.pi]:
            theta_plot = np.mod(self.theta, 2 * np.pi)
            self.theta_plot = theta_plot - 2 * np.pi * (theta_plot > np.pi)
            self.theta_min, self.theta_max = theta_lim
        else:
            print("theta_lim must be either [0,2*np.pi] or [-np.pi,np.pi].")
            return

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        fig.tight_layout()
        fig.suptitle("Drift orbits of $P_\\theta - \\theta$ and $P_\zeta - \zeta$")

        ax[0].scatter(self.theta_plot, self.Ptheta, **self.Config.drift_scatter_kw)
        ax[1].plot(self.z, self.Pzeta, **self.Config.drift_plot_kw)

        ax[0].set_xlabel("$\\theta$", **self.Config.drift_xlabel_kw)
        ax[1].set_xlabel("$\\zeta$", **self.Config.drift_xlabel_kw)

        ax[0].set_ylabel("$P_\\theta$", **self.Config.drift_ylabel_kw)
        ax[1].set_ylabel("$P_ζ$", **self.Config.drift_ylabel_kw)

        ax[1].set_ylim([-self.psip_wall, self.psip_wall])
        plt.sca(ax[0])
        # Store plot lims for contour plot
        self.drift_xlim = ax[0].get_xlim()
        self.drift_ylim = ax[0].get_ylim()

        # Set all xticks as multiples of π, and then re-set xlims (smart!)
        ticks = ["-2π", "-3π/2", "-π", "-π/2", "0", "π/2", "π", "3π/2", "2π"]
        plt.xticks(np.linspace(-2 * np.pi, 2 * np.pi, 9), ticks)
        plt.xlim(theta_lim)

    def contour_energy(
        self, scatter=True, theta_lim="auto", psi_lim="auto", levels=None
    ):
        """Draws a 2D contour plot of the Hamiltonian

        Args:
            scatter (bool, optional): Plot drift plot on top. Defaults to True.
            theta_lim (str/list, optional): "auto" uses the same limits from the
                    previous drift plots. Defaults to "auto".
            psi_lim (str/list, optional): same as theta_lim. Defaults to "auto".
            levels (int, optional): The number of contour levels. Defaults to 
                    Config setting.
        """

        # Set grid limits
        if psi_lim == "auto":  # Use ylim from drift plots
            psi_min, psi_max = self.drift_ylim
        else:
            psi_min, psi_max = psi_lim

        # Calculate Energy values
        grid_density = self.Config.contour_grid_density
        theta, psi = np.meshgrid(
            np.linspace(self.theta_min, self.theta_max, grid_density),
            np.linspace(psi_min, psi_max, grid_density),
        )
        values = self.calcW_grid(theta, psi, self.Pz0)
        span = np.array([values.min(), values.max()])

        # Create Figure
        if levels is None:  # If non is given
            levels = self.Config.contour_levels_default
        contour_kw = {
            "vmin": span[0],
            "vmax": span[1],
            "levels": levels,
            "cmap": self.Config.contour_cmap,
        }
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        ax.set_facecolor("white")

        # Contour plot
        C = ax.contourf(theta, psi, values, **contour_kw)
        ax.set_xlabel("$\\theta$")
        ax.set_ylabel("$\\psi_p\t$", rotation=0)
        plt.scatter(self.theta_plot, self.psip, **self.Config.drift_scatter_kw)
        ticks = ["-2π", "-3π/2", "-π", "-π/2", "0", "π/2", "π", "3π/2", "2π"]
        plt.xticks(np.linspace(-2 * np.pi, 2 * np.pi, 9), ticks)
        ax.set(xlim=[self.theta_min, self.theta_max], ylim=[psi_min, psi_max])
        fig.colorbar(C, ax=ax, fraction=0.03, pad=0.2)

    def plot_orbit_type_point(self):
        """Plots the particle point on the μ-Pz (normalized) plane."""

        B0 = 1  # Field magnitude on magnetic axis
        x = self.Pz0 / self.psip_wall
        y = self.mu * B0 / self.E

        plt.plot(x, y, **self.Config.orbit_point_kw)
        label = "  Particle " + f"({self.t_or_p[0]}-{self.l_or_c[0]})"
        plt.annotate(label, (x, y))

    def plot_torus2d(self, percentage=100):
        """Plots the poloidal and toroidal view of the orbit.

        Args:
            percentage (int): 0-100: the percentage of the orbit to be plotted
        """

        points = int(np.floor(self.theta.shape[0] * percentage / 100) - 1)
        self.theta_torus_plot = self.theta[:points]
        self.psi_torus_plot = self.psi[:points]
        self.z_torus_plot = self.z[:points]

        # Torus shape parameters
        scale = 4
        self.Rmin = scale * np.sqrt(2 * self.psi.min())
        # Rmax = 1.2*np.sqrt(2*self.psip.max())
        r_plot1 = self.Rmin + np.sqrt(2 * self.psi_torus_plot)
        r_plot2 = self.Rmin + np.sqrt(2 * self.psi_torus_plot) * np.cos(
            self.theta_torus_plot
        )
        self.Rmax = 1.1 * r_plot1.max()

        fig, ax = plt.subplots(1, 2, subplot_kw={"projection": "polar"})
        fig.tight_layout()

        # Torus Walls
        ax[0].scatter(
            np.linspace(0, 2 * np.pi, 1000),
            self.Rmin * np.ones(1000),
            **self.Config.torus2d_wall_kw,
        )
        ax[0].scatter(
            np.linspace(0, 2 * np.pi, 1000),
            self.Rmax * np.ones(1000),
            **self.Config.torus2d_wall_kw,
        )
        ax[1].scatter(
            np.linspace(0, 2 * np.pi, 1000),
            self.Rmin * np.ones(1000),
            **self.Config.torus2d_wall_kw,
        )
        ax[1].scatter(
            np.linspace(0, 2 * np.pi, 1000),
            self.Rmax * np.ones(1000),
            **self.Config.torus2d_wall_kw,
        )

        # Orbits
        ax[0].scatter(self.theta_torus_plot, r_plot1, **self.Config.torus2d_orbit_kw)
        ax[1].scatter(self.z_torus_plot, r_plot2, **self.Config.torus2d_orbit_kw)

        ax[0].grid(False)
        ax[1].grid(False)
        ax[0].set_xlabel("$\sqrt{2\psi} - \\theta$")
        ax[1].set_xlabel("$\sqrt{2\psi}\cos\\theta - \\zeta$")
        ax[0].tick_params(labelsize=8)
        ax[1].tick_params(labelsize=8)

    def plot_torus3d(self, percentage=100, hd=True):
        """Creates a 3d transparent torus and a part of the particle's orbit

        Args:
            percentage (int): 0-100: the percentage of the orbit to be plotted
        hd (bool):
            High definition image
        """
        points = int(np.floor(self.theta.shape[0] * percentage / 100) - 1)
        Rtorus = 1
        rtorus = 0.5

        # Cartesian
        x = (
            Rtorus + np.sqrt(2 * self.psi[:points]) * np.cos(self.theta[:points])
        ) * np.cos(self.z[:points])
        y = (
            Rtorus + np.sqrt(2 * self.psi[:points]) * np.cos(self.theta[:points])
        ) * np.sin(self.z[:points])
        z = np.sin(self.theta[:points])

        # Set dpi
        if hd:
            dpi = 900
        else:
            dpi = 300

        fig, ax = plt.subplots(
            figsize=(10, 10),
            dpi=dpi,
            subplot_kw={"projection": "3d"},
        )
        # Torus Surface
        theta_torus = np.linspace(0, 2 * np.pi, 400)
        z_torus = theta_torus
        theta_torus, z_torus = np.meshgrid(theta_torus, z_torus)
        x_torus = (Rtorus + rtorus * np.cos(theta_torus)) * np.cos(z_torus)
        y_torus = (Rtorus + rtorus * np.cos(theta_torus)) * np.sin(z_torus)
        z_torus = np.sin(theta_torus)

        # Plot z-axis
        ax.plot([0, 0], [0, 0], [-5, 5], color="k", alpha=0.4)
        # Plot wall surface
        ax.plot_surface(
            x_torus,
            y_torus,
            z_torus,
            rstride=5,
            cstride=5,
            **self.Config.torus3d_wall_kw,
        )

        ax.plot(x, y, z, **self.Config.torus3d_orbit_kw)
        ax.set_zlim([-3, 3])
        ax.set_facecolor("white")
        ax.set_box_aspect((1, 1, 0.5), zoom=1.1)
        ax.set_xlim3d(0.8 * x_torus.min(), 0.8 * x_torus.max())
        ax.set_ylim3d(0.8 * y_torus.min(), 0.8 * y_torus.max())
        ax.set_zlim3d(-3, 3)
        ax.axis("off")