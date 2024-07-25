"""
This module initializes the "Particle" class, which calculates the orbit,
orbit type, and can draw several different plots
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import Source.utils as utils
import Source.Parabolas as Parabolas


class Particle:
    """Initializes a particle.

    Supposedly there is no need to change anything here. Electric Fields and Q factors
    should be changed in the ./Functions folder.
    """

    def __init__(self, species, init_cond, mu, tspan, q, B, Efield, psi_wall):
        """Initializes particle and grabs configuration.

        Args:
            species (str): the particle species, used to later set charge and mass
                                automatically (from config.yaml)
            init_cond (np.array): 1x3 initial conditions array (later, self.init_cond
                                includes 2 more initial conditions)
            mu (float): magnetic moment
            tspan (np.array): The ODE interval, in [t0, tf, steps]
            q (object): Qfactor object that supports query methods for getting values
                                of ψ(ψ_p), ψ_p(ψ), q(ψ) and q(ψ_p)
            B (list, optional): The 3 componets of the contravariant representation
                                of the magnetic field B. Defaults to [0, 1, 0].
            E (object): Electric Field Object that supports query methods for getting
                                values for the field itself and some derivatives of
                                its potential.
            psi_wall (float, optional): The value of ψ at the wall. Better be low enough
                                so that ψ_p is lower than 0.5, which depends on the q
                                factor.
        """

        # Grab configuration
        self.Config = utils.ConfigFile()

        # Initialization
        self.species = species
        self.mass_amu = self.Config.constants[self.species + "_mass_amu"]
        self.mass_keV = self.Config.constants[self.species + "_mass_keV"]
        self.mass_kg = self.Config.constants[self.species + "_mass_kg"]
        self.charge = self.Config.constants[self.species + "_charge"]
        self.elementary_charge = self.Config.constants["elementary_charge"]
        self.Z = self.Config.constants[self.species + "_Z"]

        self.theta0 = init_cond[0]
        self.psi0 = init_cond[1]
        self.z0 = init_cond[2]
        self.Pz0 = init_cond[3]
        self.psip0 = q.psip_from_psi(self.psi0)
        self.rho0 = self.Pz0 + self.psip0  # Pz0 + psip0
        init_cond.insert(2, self.psip0)
        init_cond.insert(5, self.rho0)
        self.init_cond = np.array(init_cond)  # contains all 5
        self.tspan = tspan

        self.mu = mu
        self.q = q
        self.B = B
        self.I, self.g, self.B0 = self.B  # Β0 in Gauss
        self.Efield = Efield
        self.psi_wall = psi_wall
        self.psip_wall = self.q.psip_from_psi(self.psi_wall)
        self.r0 = np.sqrt(2 * self.psi0)
        self.r_wall = np.sqrt(2 * psi_wall)

        self.R = 1  # meters
        self.Baxis = 1
        self.B_init = 1 - self.r0 * np.cos(self.theta0)
        self.Phi_init = self.Efield.Phi_of_psi(self.psi0)

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

        # Calculate orbit type upon initialization
        self.orbit_type_str = self.orbit_type(info=True)
        print(self.__str__())

    def __str__(self):
        # grab orbit_type() results
        return self.orbit_type_str

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

        def dSdt(t, S, mu=None):
            """Sets the diff equations system to pass to scipy"""

            theta, psi, psip, z, rho = S

            # Intermediate values
            phi_der_psip, phi_der_theta = self.Efield.Phi_der_of_psi(psi)
            q_value = self.q.q_of_psi(psi)
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            r = np.sqrt(2 * psi)
            B = 1 - r * cos_theta  # B0?
            par = self.mu + rho**2 * B
            bracket1 = -par * q_value * cos_theta / r + phi_der_psip
            bracket2 = par * r * sin_theta + phi_der_theta
            D = self.g * q_value + self.I

            # Canonical Equations
            theta_dot = 1 / D * rho * B**2 + self.g / D * bracket1
            psi_dot = -self.g / D * bracket2 * self.q.q_of_psi(psi)
            psip_dot = psi_dot / self.q.q_of_psi(psi)
            rho_dot = psi_dot / (self.g * self.q.q_of_psi(psi))
            z_dot = rho * B**2 / D - self.I / D * bracket1

            return np.array([theta_dot, psi_dot, psip_dot, z_dot, rho_dot])

        sol_init = np.delete(self.init_cond, 4)  # Drop Pz0
        self.sol = odeint(dSdt, y0=sol_init, t=self.tspan, tfirst=True)

        self.theta = self.sol.T[0]
        self.psi = self.sol.T[1]
        self.psip = self.q.psip_from_psi(self.psi)
        self.z = self.sol.T[3]
        self.rho = self.sol.T[4]

        # Calculate Canonical Momenta
        self.Ptheta = self.psi + self.rho * self.I
        self.Pzeta = self.rho * self.g - self.psip

        self.calculated_orbit = True

    def orbit_type(self, info=True):
        """
        Calculates the orbit type given the initial conditions ONLY.

        Trapped/passing:
        The particle is trapped if rho vanishes, so we can
        check if rho changes sign. Since rho = (2W - 2μB)^(1/2)/B, we need only to
        check under the root.

        Confined/lost:
        (from shape page 87 i guess)
        We only have to check if the particle is in-between the 2 left parabolas.

        """

        # Calculate conversion factors
        self.conversion_factors()

        # Constants of Motion: Particle energy and Pz
        self.E = (  # Energy from initial conditions
            (self.Pz0 + self.psip0) ** 2 * self.B_init**2 / (2 * self.g**2)
            + self.mu * self.B_init
            + self.Phi_init
        )
        self.E_Joule = self.norm_to_J * self.E  # Energy in Joules
        self.E_eV = self.J_to_eV * self.E_Joule  # Energy in eV

        self.gyro_radius = np.abs(
            np.sqrt(self.mass_kg * self.E_Joule) / (self.B0 * self.Z)
        )  # ρ in m

        # Calculate Bmin and Bmax. In LAR, B decreases outwards.
        self.Bmin = 1 - np.sqrt(2 * self.psi_wall)  # "Bmin occurs at psi_wall, θ = 0"
        self.Bmax = 1 + np.sqrt(2 * self.psi_wall)  # "Bmax occurs at psi_wall, θ = π"

        # Find if trapped or passing from rho (White page 83)
        rho_min = np.sqrt(2 * self.E - 2 * self.mu * self.Bmin) / self.Bmin
        rho_max = np.sqrt(2 * self.E - 2 * self.mu * self.Bmax) / self.Bmax
        if rho_min * rho_max < 0:
            self.t_or_p = "Trapped"
        else:
            self.t_or_p = "Passing"

        # Find if lost or confined
        orbit_x = self.Pz0 / self.psip_wall
        orbit_y = self.mu / self.E
        foo = Parabolas.Orbit_parabolas(self)

        # Recalculate y by reconstructing the parabola (there might be a better way
        # to do this)
        upper_y = foo.abcs[0][0] * orbit_x**2 + foo.abcs[0][1] * orbit_x + foo.abcs[0][2]
        lower_y = foo.abcs[1][0] * orbit_x**2 + foo.abcs[1][1] * orbit_x + foo.abcs[1][2]

        if orbit_y < upper_y and orbit_y > lower_y:
            self.l_or_c = "Lost"
        else:
            self.l_or_c = "Confined"

        # String to return to __str__()
        self.orbit_type_str = (
            "Constants of motion:\n"
            + "\tParticle Energy (normalized):\tE = {:e}\n".format(self.E)
            + "\tParticle Energy (keV):\t\tE = {:e} eV\n".format(self.E_eV)
            + f"\tToroidal Momenta:\t\tPζ = {self.Pz0}\n\n"
            + "Other Quantities:\n"
            + f'\tParticle of Species:\t\t"{self.species}"\n'
            + f"\tOrbit Type:\t\t\t{self.t_or_p} - {self.l_or_c}\n"
            + f"\tMajor Radius:\t\t\tR = {self.R} meters\n"
            + "\tTime unit:\t\t\tω = {:e} Hz \n".format(self.w0)
            + "\tGyro radius: \t\t\tρ = {:e} cm \n".format(self.gyro_radius * 100)
        )

        if info:
            return self.orbit_type_str

    def conversion_factors(self):  # BETA
        e = self.charge  # 1.6*10**(-19)C
        m_kg = self.mass_kg
        B = self.B0  # Gauss
        R = self.R  # meters

        self.w0 = np.abs(e) * 10 ** (-4) * B / (m_kg)  # s^-1
        self.E_unit = m_kg * self.w0**2 * R**2  # Multiply normalised energy to get J

        # Conversion Factors
        self.norm_to_J = self.E_unit
        self.J_to_eV = 1 / self.elementary_charge
        self.norm_to_eV = self.norm_to_J * self.J_to_eV
        self.eV_to_norm = 1 / self.norm_to_eV
        self.kV_to_V = 1000
        self.V_to_eV = self.elementary_charge
        self.kV_to_eV = self.kV_to_V * self.V_to_eV
        self.kV_to_keV = self.kV_to_eV / 1000
        self.kV_to_norm = self.kV_to_eV * self.eV_to_norm

    def calcW_grid(self, theta, psi, Pz, contour_Phi=True, units=True):
        """Returns a single value or a grid of the calculated Hamiltonian.

        Depending on the value of contour_Phi, it can include the Electric
        Potential energy or not
        """

        r = np.sqrt(2 * psi)
        B = 1 - r * np.cos(theta)
        psip = self.q.psip_from_psi(psi)

        W_magnetic = (Pz + psip) ** 2 * B**2 / (2 * self.g**2) + self.mu * B

        if contour_Phi:
            Phi = self.Efield.Phi_of_psi(psi)
            # Phi *= self.kV_to_norm
        else:
            Phi = 0 * theta  # Grid of zeros

        W = W_magnetic + Phi  # all normalized

        if units == "eV":
            W *= self.norm_to_eV
        elif units == "keV":
            W *= self.norm_to_eV / 1000

        return W

    def plot_electric(self, q_plot=False, zoom=None):
        """Plots the electric field, potential, and q factor

        Args:
            q_plot (bool, optional): Plot q factor. Defaults to False.
            zoom (list, optional): zoom to specific area in the x-axis of the electric
                                field and potential plots. Defaults to None.
        """

        psi = np.linspace(0, 1.1 * self.psi_wall, 1000)
        Er = self.Efield.Er_of_psi(psi)
        Phi = self.Efield.Phi_of_psi(psi)

        if q_plot:
            fig_dim = (2, 2)
            figsize = (14, 8)
        else:
            fig_dim = (1, 2)
            figsize = (14, 4)

        fig, ax = plt.subplots(fig_dim[0], fig_dim[1], figsize=figsize, dpi=200)
        fig.subplots_adjust(hspace=0.3)

        # Radial E field
        ax[0][0].plot(psi / self.psi_wall, Er, color="b", linewidth=3)
        ax[0][0].plot([1, 1], [Er.min(), Er.max()], color="r", linewidth=3)
        ax[0][0].set_xlabel("$\psi/\psi_{wall}$")
        ax[0][0].set_ylabel("$E_r$ [kV/m]")
        ax[0][0].set_title("Radial electric field [kV/m]", c="b")

        # Electric Potential
        ax[0][1].plot(psi / self.psi_wall, Phi, color="b", linewidth=3)
        ax[0][1].plot([1, 1], [Phi.min(), Phi.max()], color="r", linewidth=3)
        ax[0][1].set_xlabel("$\psi/\psi_{wall}$")
        ax[0][1].set_ylabel("$Φ_r$ [kV]")
        ax[0][1].set_title("Electric Potential [kV]", c="b")

        if zoom is not None:
            ax[0][0].set_xlim(zoom)
            ax[0][1].set_xlim(zoom)

        if not q_plot:
            return

        # q(ψ)
        y1 = self.q.q_of_psi(psi)
        ax[1][0].plot(psi / self.psi_wall, y1, color="b", linewidth=3)
        ax[1][0].plot([1, 1], [y1.min(), y1.max()], color="r", linewidth=3)

        ax[1][0].set_xlabel("$\psi/\psi_{wall}$")
        ax[1][0].set_ylabel("$q(\psi)$", rotation=0)
        ax[1][0].set_title("$\\text{q factor }q(\psi)$", c="b")

        # ψ_π(ψ)
        y2 = self.q.psip_from_psi(psi)
        ax[1][1].plot(psi / self.psi_wall, y2, color="b", linewidth=3)
        ax[1][1].plot([1, 1], [y2.min(), y2.max()], color="r", linewidth=3)
        ax[1][1].set_xlabel("$\psi/\psi_{wall}$")
        ax[1][1].set_ylabel("$\psi_p(\psi)$", rotation=0)
        ax[1][1].set_title("$\psi_p(\psi)$", c="b")

    def plot_time_evolution(self, percentage=100):
        """
        Plots the time evolution of the dynamical variabls and
        canonical momenta.

        Args:
        percentage (int): 0-100: the percentage of the orbit to be plotted
        """

        points = int(np.floor(self.theta.shape[0] * percentage / 100) - 1)

        # Plotting
        fig, ax = plt.subplots(7, 1, figsize=(10, 8), sharex=True, dpi=300)
        fig.tight_layout()
        ax[0].set_title("Time evolution of dynamical variables", c="b")
        ax[5].set_title("Time evolution of canonical momenta", c="b")

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

    def plot_drift(self, theta_lim):
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

    def plot_Ptheta_drift(self, theta_lim):
        """Draws θ - P_θ plot.

        Args:
            theta_lim (np.array): x-axis limits
        """

        # Set theta lim. Mods all thetas to 2π
        theta_min, theta_max = theta_lim
        self.theta_plot = utils.theta_plot(self.theta, theta_lim)

        plt.scatter(self.theta_plot, self.Ptheta, **self.Config.drift_scatter_kw, zorder=1)
        plt.set_xlabel("$\\theta$", **self.Config.drift_xlabel_kw)
        plt.set_ylabel("$P_\\theta$", **self.Config.drift_ylabel_kw)

        # Set all xticks as multiples of π, and then re-set xlims (smart!)
        ticks = ["-2π", "-3π/2", "-π", "-π/2", "0", "π/2", "π", "3π/2", "2π"]
        plt.xticks(np.linspace(-2 * np.pi, 2 * np.pi, 9), ticks)
        plt.xlim(theta_lim)

    def contour_energy(
        self,
        theta_lim,
        psi_lim="auto",
        plot_drift=True,
        contour_Phi=True,
        units="keV",
        levels=None,
    ):
        """Draws a 2D contour plot of the Hamiltonian

        Can also plot the current particle's θ-Pθ drift. Should be False when
        running with multiple initial conditions.
        Args:
            scatter (bool, optional): Plot drift plot on top. Defaults to True.
            theta_lim (str/list, optional): "auto" uses the same limits from the
                    previous drift plots. Defaults to "auto".
            psi_lim (str/list, optional): same as theta_lim. Defaults to "auto".
            plot_drift (bool, optional): Whether or not to plot θ=Ρθ drift on top.
            contour_Phi (bool, optional): Whether or not to add the Φ term in the
                    energy contour.
            units (str, optional): The units in which energies are displayed. Must
                    be either "normal", "eV", or "keV".
            levels (int, optional): The number of contour levels. Defaults to
                    Config setting.
        """

        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)

        # Set theta lim. Mods all thetas to 2π
        self.theta_min, self.theta_max = theta_lim
        self.theta_plot = utils.theta_plot(self.theta, theta_lim)

        if plot_drift:
            ax.scatter(
                self.theta_plot,
                self.Ptheta / self.psi_wall,
                **self.Config.drift_scatter_kw,
                zorder=2,
            )

        if units == "normal":
            label = "E (normalized)"
            E_label = self.E
        elif units == "eV":
            label = "E (eV)"
            E_label = self.E_eV
        elif units == "keV":
            label = "E (keV)"
            E_label = self.E_eV / 1000
        else:
            print('units must be either "normal", "eV" or "keV"')
            return

        # Set psi limits (Normalised to psi_wall)
        if psi_lim == "auto":
            psi_diff = self.psi.max() - self.psi.min()
            psi_mid = (self.psi.max() + self.psi.min()) / 2
            psi_lower = max(0, psi_mid - 0.6 * psi_diff)
            psi_higher = psi_mid + 0.6 * psi_diff
            psi_lim = np.array([psi_lower, psi_higher])
            psi_lim /= self.psi_wall
        psi_min = psi_lim[0]
        psi_max = psi_lim[1]

        # Calculate Energy values
        grid_density = self.Config.contour_grid_density
        theta, psi = np.meshgrid(
            np.linspace(self.theta_min, self.theta_max, grid_density),
            np.linspace(psi_min * self.psi_wall, psi_max * self.psi_wall, grid_density),
        )
        values = self.calcW_grid(theta, psi, self.Pz0, contour_Phi, units)
        span = np.array([values.min(), values.max()])

        # Create Figure
        if levels is None:  # If non is given
            levels = self.Config.contour_levels_default
        contour_kw = {
            "vmin": span[0],
            "vmax": span[1],
            "levels": levels,
            "cmap": self.Config.contour_cmap,
            "zorder": 1,
        }

        # Contour plot
        C = ax.contourf(theta, psi / self.psi_wall, values, **contour_kw)
        ax.set_xlabel("$\\theta$")
        ax.set_ylabel("$\\psi/\\psi_{wall}\t$", rotation=90)
        ticks = ["-2π", "-3π/2", "-π", "-π/2", "0", "π/2", "π", "3π/2", "2π"]
        plt.xticks(np.linspace(-2 * np.pi, 2 * np.pi, 9), ticks)
        ax.set(xlim=[self.theta_min, self.theta_max], ylim=psi_lim)
        ax.set_facecolor("white")
        cbar = fig.colorbar(C, ax=ax, fraction=0.03, pad=0.2, label=label)
        # Draw a small dash over the colorbar indicating the particle's energy level
        cbar.ax.plot([0, 1], [E_label, E_label], linestyle="-", c="r", zorder=3)

    def plot_orbit_type_point(self):  # Needs checking
        """Plots the particle point on the μ-Pz (normalized) plane."""

        x = self.Pz0 / self.psip_wall
        y = self.mu * self.B0 / self.E_eV

        plt.plot(x, y, **self.Config.orbit_point_kw)
        label = "  Particle " + f"({self.t_or_p[0]}-{self.l_or_c[0]})"
        plt.annotate(label, (x, y), color="b")

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
        self.r_span = [
            np.sqrt(2 * self.psi_torus_plot.min()),
            np.sqrt(2 * self.psi_torus_plot.max()),
        ]
        self.R = 2 * (self.r_span[1] + self.r_span[0]) / 2
        # rtorus = self.Rout - self.R

        self.Rin = self.R - self.r_span[0]
        self.Rout = self.R + self.r_span[1]

        r_plot1 = np.sqrt(2 * self.psi_torus_plot)
        r_plot2 = self.R + np.sqrt(2 * self.psi_torus_plot) * np.cos(self.theta_torus_plot)

        fig, ax = plt.subplots(1, 2, figsize=(8, 5), subplot_kw={"projection": "polar"})
        fig.tight_layout()

        # Torus Walls
        ax[0].scatter(
            np.linspace(0, 2 * np.pi, 1000),
            1.05 * self.r_span[1] * np.ones(1000),
            **self.Config.torus2d_wall_kw,
        )
        ax[1].scatter(
            np.linspace(0, 2 * np.pi, 1000),
            0.90 * self.Rin * np.ones(1000),
            **self.Config.torus2d_wall_kw,
        )
        ax[1].scatter(
            np.linspace(0, 2 * np.pi, 1000),
            1.05 * self.Rout * np.ones(1000),
            **self.Config.torus2d_wall_kw,
        )

        # Orbits
        ax[0].scatter(self.theta_torus_plot, r_plot1, **self.Config.torus2d_orbit_kw, zorder=-1)
        ax[1].scatter(self.z_torus_plot, r_plot2, **self.Config.torus2d_orbit_kw, zorder=-1)

        ax[0].set_ylim(bottom=0)
        ax[1].set_ylim(bottom=0)
        ax[0].grid(False)
        ax[1].grid(False)
        ax[0].set_title("Toroidal View", c="b")
        ax[1].set_title("Top-Down View", c="b")
        ax[0].set_xlabel("$\sqrt{2\psi} - \\theta$")
        ax[1].set_xlabel("$\sqrt{2\psi}\cos\\theta - \\zeta$")
        ax[0].tick_params(labelsize=8)
        ax[1].tick_params(labelsize=8)

    def plot_torus3d(self, percentage=100, hd=True, bold=False, white_background=True):
        """Creates a 3d transparent torus and a part of the particle's orbit

        Args:
            percentage (int): 0-100: the percentage of the orbit to be plotted
        hd (bool):
            High definition image
        """
        points = int(np.floor(self.theta.shape[0] * percentage / 100) - 1)
        psi_plot = self.psi[:points]
        theta_plot = self.theta[:points]
        z_plot = self.z[:points]

        Rtorus = self.R
        rtorus = 1.1 * np.sqrt(2 * psi_plot.max())

        # Cartesian
        x = (Rtorus + np.sqrt(2 * psi_plot) * np.cos(theta_plot)) * np.cos(z_plot)
        y = (Rtorus + np.sqrt(2 * psi_plot) * np.cos(theta_plot)) * np.sin(z_plot)
        z = np.sin(theta_plot)

        # Set dpi
        if hd:
            dpi = 900
        else:
            dpi = 300

        # plt.rcParams["pad_inches"] = 0

        fig, ax = plt.subplots(
            dpi=dpi,
            subplot_kw={"projection": "3d"},
            **{"figsize": (10, 6), "frameon": False},
        )
        ax.set_axis_off()

        # Torus Surface
        theta_torus = np.linspace(0, 2 * np.pi, 400)
        z_torus = theta_torus
        theta_torus, z_torus = np.meshgrid(theta_torus, z_torus)
        x_torus = (Rtorus + rtorus * np.cos(theta_torus)) * np.cos(z_torus)
        y_torus = (Rtorus + rtorus * np.cos(theta_torus)) * np.sin(z_torus)
        z_torus = np.sin(theta_torus)

        # Plot z-axis
        ax.plot([0, 0], [0, 0], [-8, 6], color="k", alpha=0.4, linewidth=0.5)
        # Plot wall surface
        ax.plot_surface(
            x_torus,
            y_torus,
            z_torus,
            rstride=3,
            cstride=3,
            **self.Config.torus3d_wall_kw,
        )

        custom_kw = self.Config.torus3d_orbit_kw.copy()
        if bold:
            custom_kw["alpha"] = 1
            custom_kw["linewidth"] *= 3

        if white_background:
            ax.set_facecolor("white")

        ax.plot(x, y, z, **custom_kw, zorder=1)
        # ax.set_zlim([-8, 8])
        ax.set_box_aspect((1, 1, 0.5), zoom=1.1)
        ax.set_xlim3d(0.8 * x_torus.min(), 0.8 * x_torus.max())
        ax.set_ylim3d(0.8 * y_torus.min(), 0.8 * y_torus.max())
        ax.set_zlim3d(-3, 3)
