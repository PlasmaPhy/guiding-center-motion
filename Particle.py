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
import Functions.Qfactors as Qfactors

reload(Parabolas)  # for debugging


class Particle:
    """Initializes a particle.

    Supposedly there is no need to change anything here. Electric Fields and Q factors
    should be changed in the ./Functions folder.
    """

    def __init__(
        self,
        species,
        init_cond,
        mu,
        tspan,
        q=Qfactors.Unity,
        B=[0, 1, 0],
        Efield=None,
        psi_wall=0.3,
    ):  # Ready to commit
        """Initializes particle and grabs configuration.

        Args:
            species (str): the particle species
            init_cond (np.array): 1x3 initial conditions array (self.init_cond includes
                                2 more initial conditions)
            mu (float): magnetic moment
            tspan (np.array): The ODE interval, in [t0, tf, steps]
            q (object): Qfactor object that supports query methods for getting values
                                of ψ/ψ_p and q
            B (list, optional): The 3 componets of the contravariant representation
                                of the magnetic field B. Defaults to [0, 1, 0].
            E (object): Electric Field Object that supports query methods for getting
                                values for the field itself and some derivatives of
                                its potential.
            psi_wall (float, optional): The value of ψ at the wall. Better be low enough
                                so that ψ_p is lower than 0.5, which depends on the q
                                factor. Defaults to 0.3.
        """

        # Initialization
        self.species = species
        psip0 = q.psip_from_psi(init_cond[1])
        rho0 = init_cond[3] + psip0  # Pz0 + psip0
        init_cond.insert(2, psip0)
        init_cond.insert(5, rho0)
        self.init_cond = np.array(init_cond)
        self.mu = mu
        self.q = q
        self.B = B
        self.I, self.g, self.delta = self.B
        self.Efield = Efield
        self.psi_wall = psi_wall
        self.psip_wall = self.q.psip_from_psi(self.psi_wall)
        self.psi_wall = np.sqrt(2 * psi_wall)
        self.tspan = tspan
        # psi_p > 0.5 warning
        if self.psip_wall >= 0.5:
            print(
                f"WARNING: psip_wall = {self.psip_wall} >= 0,5."
                + "Parabolas and other stuff will probably not work"
            )

        # Logic variables
        self.calculated_orbit = False
        self.t_or_p = "Unknown"
        self.l_or_c = "Unknown"

        # Grab configuration
        self.Config = utils.Config_file()

        # Run
        self.orbit()

    def __str__(self):  # Ready to commit
        string = (
            f'Particle of Species:\t "{self.species}"\n'
            + f"Calculated orbit:\t {self.calculated_orbit}\n\n"
        )
        # Also grab orbit_type() results
        string += self.orbit_type(info=True)
        return string

    def orbit(self):  # Ready to commit
        """Calculates the orbit of the particle.

        Calculates a 2D numpy array, containing the 4 time evolution vectors of
        each dynamical variable (θ, ψ_p, ζ, ρ). Afterwards, it calculates the
        canonical momenta (P_θ, P_ζ).
        Orbit is stored in "self"
        """

        # Check if orbit has been already calculated
        if self.calculated_orbit:
            return

        def dSdt_psi(t, S, mu=None):
            """Sets the diff equations system to pass to scipy"""

            theta, psi, psip, z, rho = S

            # Intermediate values
            r = np.sqrt(2 * psi)
            if self.Efield is None:
                phi_der_psip = phi_der_theta = 0
            else:
                phi_der_psip, phi_der_theta = self.Efield.orbit(r)

            q_value = self.q.q_of_psi(psi)
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            B = 1 - r * cos_theta
            par = self.mu + rho**2 * B
            bracket1 = -par * q_value * cos_theta / r + phi_der_psip
            bracket2 = par * r * sin_theta + phi_der_theta
            D = self.g * q_value + self.I

            # Canonical Equation
            theta_dot = 1 / D * rho * B**2 + self.g / D * bracket1
            psi_dot = -self.g / D * bracket2 * self.q.q_of_psi(psi)
            psip_dot = psi_dot / self.q.q_of_psi(psi)
            rho_dot = psi_dot / (self.g * self.q.q_of_psi(psi))
            z_dot = rho * B**2 / D - self.I / D * bracket1

            return np.array([theta_dot, psi_dot, psip_dot, z_dot, rho_dot])

        sol_init = np.delete(self.init_cond, 4)  # Drop Pz0
        self.sol = odeint(dSdt_psi, y0=sol_init, t=self.tspan, tfirst=True)

        self.theta = self.sol.T[0]
        self.psi = self.sol.T[1]
        self.psip = self.q.psip_from_psi(self.psi)
        self.z = self.sol.T[3]
        self.rho = self.sol.T[4]

        self.calculated_orbit = True

        # Calculate Canonical Momenta
        self.Ptheta = self.psi + self.rho * self.I
        self.Pzeta = self.rho * self.g - self.psip

    def orbit_type(self, info=True):  # Ready to commit
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
        self.theta0, self.psi0, self.psip0, self.z0, self.Pz0, self.rho0 = self.init_cond
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
        foo = Parabolas.Orbit_parabolas(self.E, self.q, self.B, self.Efield, self.psi_wall)

        # Recalculate y by reconstructing the parabola (there might be a better way
        # to do this)
        upper_y = foo.abcs[0][0] * particle_x**2 + foo.abcs[0][1] * particle_x + foo.abcs[0][2]
        lower_y = foo.abcs[1][0] * particle_x**2 + foo.abcs[1][1] * particle_x + foo.abcs[1][2]

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

    def calcW_grid(self, theta, psi, Pz, contour_Efield=True):  # Ready to commit
        """Returns a single value or a grid of the calculated Hamiltonian.

        Depending on the value of contour_Efield, it can include the Electric
        Potential energy or not
        """

        r = np.sqrt(2 * psi)
        B = 1 - r * np.cos(theta)
        psip = self.q.psip_from_psi(psi)

        if contour_Efield:
            if self.Efield is None:
                Phi = 0 * theta  # Grid of zeros
                print("foo")
            else:
                Phi = self.Efield.Phi_of_psi(psi)
                # print(Phi)
                print("bar")
        else:
            Phi = 0 * theta  # Grid of zeros
            print("foo2")

        return (Pz + psip) ** 2 * B**2 / (2 * self.g**2) + self.mu * B + Phi

    def plot_electric(self):  # BETA

        if self.Efield is None:
            print("No electric field")
            return

        psi = np.linspace(0, 1.1 * self.psi_wall, 1000) / self.psi_wall
        Er = self.Efield.Er_of_psi(psi)
        Phi = self.Efield.Phi_of_psi(psi)

        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        ax[0].plot(psi, Er, color="b", linewidth=3)
        ax[0].plot([1, 1], [Er.min(), Er.max()], color="r", linewidth=3)
        ax[0].set_xlabel("$r/r_{wall}$")
        ax[0].set_ylabel("$E_r$")
        ax[0].set_title("Radial electric field")

        ax[1].plot(psi, Phi, color="b", linewidth=3)
        ax[1].plot([1, 1], [Phi.min(), Phi.max()], color="r", linewidth=3)
        ax[1].set_xlabel("$r/r_{wall}$")
        ax[1].set_ylabel("$Φ_r$")
        ax[1].set_title("Electric Potential")

    def plot_time_evolution(self, percentage=100):  # Ready to commit
        """
        Plots the time evolution of the dynamical variabls and
        canonical momenta.

        Args:
        percentage (int): 0-100: the percentage of the orbit to be plotted
        """

        points = int(np.floor(self.theta.shape[0] * percentage / 100) - 1)

        # Plotting
        fig, ax = plt.subplots(7, 1, figsize=(10, 8), sharex=True)
        fig.tight_layout()
        ax[0].title.set_text("Time evolution of dynamical variables")
        ax[5].title.set_text("Time evolution of canonical momenta")

        ax[0].scatter(self.tspan[:points], self.theta[:points], **self.Config.time_scatter_kw)
        ax[1].scatter(self.tspan[:points], self.z[:points], **self.Config.time_scatter_kw)
        ax[2].scatter(self.tspan[:points], self.psi[:points], **self.Config.time_scatter_kw)
        ax[3].scatter(self.tspan[:points], self.psip[:points], **self.Config.time_scatter_kw)
        ax[4].scatter(self.tspan[:points], self.rho[:points], **self.Config.time_scatter_kw)
        ax[5].scatter(self.tspan[:points], self.Ptheta[:points], **self.Config.time_scatter_kw)
        ax[6].scatter(self.tspan[:points], self.Pzeta[:points], **self.Config.time_scatter_kw)

        ax[0].set_ylabel("$\\theta(t)$\t", **self.Config.time_ylabel_kw)
        ax[1].set_ylabel("$\\zeta(t)$\t", **self.Config.time_ylabel_kw)
        ax[2].set_ylabel("$\\psi(t)$\t", **self.Config.time_ylabel_kw)
        ax[3].set_ylabel("$\\psi_p(t)$\t", **self.Config.time_ylabel_kw)
        ax[4].set_ylabel("$\\rho(t)$\t", **self.Config.time_ylabel_kw)
        ax[5].set_ylabel("$P_\\theta(t)$\t\t", **self.Config.time_ylabel_kw)
        ax[6].set_ylabel("$P_\\zeta(t)$\t", **self.Config.time_ylabel_kw)
        ax[6].set_ylim([-self.psip_wall, self.psip_wall])

        plt.xlabel("$t$")

    def plot_drift(self, theta_lim):  # Ready to commit
        """Draws 2 plots: 1] θ-P_θ and 2] ζ-P_ζ

        Args:
            theta_lim (np.array): x-axis limits
        """

        # Set theta lim. Mods all thetas to 2π
        theta_min, theta_max = theta_lim
        self.theta_plot = utils.theta_plot(self.theta, theta_lim)

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
        self,
        scatter=True,
        theta_lim="auto",
        psi_lim="auto",
        contour_Efield=True,
        levels=None,
    ):  # Ready to commit
        """Draws a 2D contour plot of the Hamiltonian

        Args:
            scatter (bool, optional): Plot drift plot on top. Defaults to True.
            theta_lim (str/list, optional): "auto" uses the same limits from the
                    previous drift plots. Defaults to "auto".
            psi_lim (str/list, optional): same as theta_lim. Defaults to "auto".
            levels (int, optional): The number of contour levels. Defaults to
                    Config setting.
        """
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)

        # Set theta lim. Mods all thetas to 2π
        self.theta_min, self.theta_max = theta_lim
        self.theta_plot = utils.theta_plot(self.theta, theta_lim)

        ax.scatter(self.theta_plot, self.Ptheta, **self.Config.drift_scatter_kw)

        # Set psi limits
        if psi_lim == "auto":  # Use ylim from drift plots
            psi_min, psi_max = ax.get_ylim()
        else:
            psi_min, psi_max = psi_lim

        # Calculate Energy values
        grid_density = self.Config.contour_grid_density
        theta, psi = np.meshgrid(
            np.linspace(self.theta_min, self.theta_max, grid_density),
            np.linspace(psi_min, psi_max, grid_density),
        )
        values = self.calcW_grid(theta, psi, self.Pz0, contour_Efield)
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

        ax.set_facecolor("white")

        # Contour plot
        C = ax.contourf(theta, psi, values, **contour_kw)
        ax.set_xlabel("$\\theta$")
        ax.set_ylabel("$\\psi\t$", rotation=0)
        plt.scatter(self.theta_plot, self.Ptheta, **self.Config.drift_scatter_kw)
        ticks = ["-2π", "-3π/2", "-π", "-π/2", "0", "π/2", "π", "3π/2", "2π"]
        plt.xticks(np.linspace(-2 * np.pi, 2 * np.pi, 9), ticks)
        ax.set(xlim=[self.theta_min, self.theta_max], ylim=[psi_min, psi_max])
        fig.colorbar(C, ax=ax, fraction=0.03, pad=0.2)

    def plot_orbit_type_point(self):  # Ready to commit
        """Plots the particle point on the μ-Pz (normalized) plane."""

        B0 = 1  # Field magnitude on magnetic axis
        x = self.Pz0 / self.psip_wall
        y = self.mu * B0 / self.E

        plt.plot(x, y, **self.Config.orbit_point_kw)
        label = "  Particle " + f"({self.t_or_p[0]}-{self.l_or_c[0]})"
        plt.annotate(label, (x, y))

    def plot_torus2d(self, percentage=100):  # Ready to commit
        """Plots the poloidal and toroidal view of the orbit.

        Args:
            percentage (int): 0-100: the percentage of the orbit to be plotted
        """

        points = int(np.floor(self.theta.shape[0] * percentage / 100) - 1)
        self.theta_torus_plot = self.theta[:points]
        self.psi_torus_plot = self.psi[:points]
        self.z_torus_plot = self.z[:points]

        # Torus shape parameters
        scale = 1
        self.Rmin = scale * np.sqrt(2 * self.psi).min()
        # Rmax = 1.2*np.sqrt(2*self.psip.max())
        r_plot1 = 3 * self.Rmin + np.sqrt(2 * self.psi_torus_plot)
        r_plot2 = 3 * self.Rmin + np.sqrt(2 * self.psi_torus_plot) * np.cos(self.theta_torus_plot)
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
        ax[0].set_title("Toroidal View")
        ax[1].set_title("Top-Down View")
        ax[0].set_xlabel("$\sqrt{2\psi} - \\theta$")
        ax[1].set_xlabel("$\sqrt{2\psi}\cos\\theta - \\zeta$")
        ax[0].tick_params(labelsize=8)
        ax[1].tick_params(labelsize=8)

    def plot_torus3d(self, percentage=100, hd=True, full_alpha=False):  # Ready to commit
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
        x = (Rtorus + np.sqrt(2 * self.psi[:points]) * np.cos(self.theta[:points])) * np.cos(
            self.z[:points]
        )
        y = (Rtorus + np.sqrt(2 * self.psi[:points]) * np.cos(self.theta[:points])) * np.sin(
            self.z[:points]
        )
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
