""" This module initialized the Plot class, which is a component of the
composite class ``Particle``, and contains all the plotting-related methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Literal
from matplotlib.patches import Rectangle
from .parabolas import Construct
from . import utils


class Plot:
    r"""Component of class ``Particle``. Contains all the plotting-related methods."""

    def __init__(self, cwp):
        r"""Copies attributes from cwp to self.

        The instance itself is automatically initialized internally by the Particle
        class, and only its methods are to be called by the user, as
        ``cwp.plot.<method()>``.

        Args:
            cwp (Particle): The Current Working Particle
        """
        self.__dict__ = dict(cwp.__dict__)

    def tokamak_profile(self, zoom: list = [0, 1.1]):
        r"""Plots the electric field, potential, and q factor,
        with respect to :math:`\psi/\psi_{wall}`.

        Args:
            zoom (list, optional): zoom to specific area in the x-axis of the electric field
                and potential plots. Defaults to [0, 1.1].
        """

        psi = np.linspace(0, 1.1 * self.psi_wall, 1000)
        Er = self.Efield.Er_of_psi(psi)
        Phi = self.Efield.Phi_of_psi(psi)

        if np.abs(Er).max() > 1000:  # If in kV
            Er /= 1000
            Phi /= 1000
            E_ylabel = "$E_r$ [kV/m]"
            Phi_ylabel = "$Φ_r$ [kV]"
        else:  # If in V
            E_ylabel = "$E_r$ [V/m]"
            Phi_ylabel = "$Φ_r$ [V]"

        # fig, ax = plt.subplots(3, 2, figsize=(14, 8), dpi=200)
        # fig.subplots_adjust(hspace=0.3)

        plt.figure(dpi=200, figsize=(10, 8))
        plt.subplots_adjust(hspace=0.6)
        ax_phi = plt.subplot(321)
        ax_E = plt.subplot(322)
        ax_q1 = plt.subplot(323)
        ax_q2 = plt.subplot(324)

        # Radial E field
        ax_phi.plot(psi / self.psi_wall, Er, color="b", linewidth=1.5)
        ax_phi.plot([1, 1], [Er.min(), Er.max()], color="r", linewidth=1.5)
        ax_phi.set_xlabel(r"$\psi/\psi_{wall}$")
        ax_phi.set_ylabel(E_ylabel)
        ax_phi.set_title("Radial electric field [kV/m]", c="b")

        # Electric Potential
        ax_E.plot(psi / self.psi_wall, Phi, color="b", linewidth=1.5)
        ax_E.plot([1, 1], [Phi.min(), Phi.max()], color="r", linewidth=1.5)
        ax_E.set_xlabel(r"$\psi/\psi_{wall}$")
        ax_E.set_ylabel(Phi_ylabel)
        ax_E.set_title("Electric Potential [kV]", c="b")

        ax_phi.set_xlim(zoom)
        ax_E.set_xlim(zoom)

        # q(ψ)
        y1 = self.q.q_of_psi(psi)
        if type(y1) is int:  # if q = Unity
            y1 *= np.ones(psi.shape)
        ax_q1.plot(psi / self.psi_wall, y1, color="b", linewidth=1.5)
        ax_q1.plot([1, 1], [y1.min(), y1.max()], color="r", linewidth=1.5)
        ax_q1.set_xlabel(r"$\psi/\psi_{wall}$")
        ax_q1.set_ylabel(r"$q(\psi)$")
        ax_q1.set_title(r"$\text{q factor }q(\psi)$", c="b")

        # ψ_π(ψ)
        y2 = self.q.psip_of_psi(psi)
        ax_q2.plot(psi / self.psi_wall, y2, color="b", linewidth=1.5)
        ax_q2.plot([1, 1], [y2.min(), y2.max()], color="r", linewidth=1.5)
        ax_q2.set_xlabel(r"$\psi/\psi_{wall}$")
        ax_q2.set_ylabel(r"$\psi_p(\psi)$")
        ax_q2.set_title(r"$\psi_p(\psi)$", c="b")

        # Magnetic Field
        ax_B = plt.subplot(212, projection="polar")
        box = ax_B.get_position()
        box.y0 = box.y0 - 0.15
        box.y1 = box.y1 - 0.15
        ax_B.set_position(box)
        ax_B.set_title("LAR Magnetic Field Profile", c="b")
        ax_B.set_rlabel_position(30)
        ax_B.set_yticks([0, self.a])

        rs = np.linspace(0, self.a, 100)
        thetas = np.linspace(0, 2 * np.pi, 100)
        r, theta = np.meshgrid(rs, thetas)
        B = self.Bfield.B(r, theta)
        ax_B.contourf(theta, r, B, levels=100, cmap="winter")

    def time_evolution(self, percentage: int = 100):
        r"""Plots the time evolution of all the dynamical variables and
        canonical momenta.

        Args:
            percentage (int, optional): The percentage of the orbit to be plotted.
                Defaults to 100.
        """

        if percentage < 1 or percentage > 100:
            percentage = 100
            print("Invalid percentage. Plotting the whole thing.")

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

        ax[0].set_ylabel(r"$\theta(t)$", **self.Config.time_ylabel_kw)
        ax[1].set_ylabel(r"$\zeta(t)$", **self.Config.time_ylabel_kw)
        ax[2].set_ylabel(r"$\psi(t)$", **self.Config.time_ylabel_kw)
        ax[3].set_ylabel(r"$\psi_p(t)$", **self.Config.time_ylabel_kw)
        ax[4].set_ylabel(r"$\rho(t)$", **self.Config.time_ylabel_kw)
        ax[5].set_ylabel(r"$P_\theta(t)$", **self.Config.time_ylabel_kw)
        ax[6].set_ylabel(r"$P_\zeta(t)$", **self.Config.time_ylabel_kw)
        ax[6].set_ylim([-self.psip_wall, self.psip_wall])

        plt.xlabel("$t$")

    def drifts(self, theta_lim: list = [-np.pi, np.pi]):
        r"""Draws 2 plots: 1] :math:`\theta-P_\theta`
        and 2] :math:`\zeta-P_\zeta`.

        Args:
            theta_lim (list, optional): Plot xlim. Must be either [0,2π] or [-π,π].
                Defaults to [-π,π].
        """

        # Set theta lim. Mods all thetas to 2π
        theta_min, theta_max = theta_lim
        theta_plot = utils.theta_plot(self.theta, theta_lim)

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        fig.tight_layout()
        fig.suptitle(r"Drift orbits of $P_\theta - \theta$ and $P_\zeta - \zeta$")

        ax[0].scatter(theta_plot, self.Ptheta, **self.Config.drift_scatter_kw)
        ax[1].plot(self.z, self.Pzeta, **self.Config.drift_plot_kw)

        ax[0].set_xlabel(r"$\theta$", **self.Config.drift_xlabel_kw)
        ax[1].set_xlabel(r"$\zeta$", **self.Config.drift_xlabel_kw)

        ax[0].set_ylabel(r"$P_\theta$", **self.Config.drifts_ylabel_kw)
        ax[1].set_ylabel(r"$P_ζ$", **self.Config.drifts_ylabel_kw)

        ax[1].set_ylim([-self.psip_wall, self.psip_wall])

        # Set all xticks as multiples of π, and then re-set xlims (smart!)
        ticks = ["-2π", "-3π/2", "-π", "-π/2", "0", "π/2", "π", "3π/2", "2π"]
        ax[0].set_xticks(np.linspace(-2 * np.pi, 2 * np.pi, 9), ticks)
        ax[0].set_xlim(theta_lim)

    def drift(
        self, angle: Literal["zeta", "theta"] = "theta", lim: list = [-np.pi, np.pi], **kwargs
    ):
        r"""Draws :math:`\theta - P_\theta` plot.

        This method is called internally by ``countour_energy()``
        as well.

        Args:
            theta_lim (list, optional): Plot xlim. Must be either [0,2π] or [-π,π].
                Defaults to [-π,π].
            ax (pyplot ax, optional): The pyplot ``ax`` object to plot upon.
        """

        canvas = kwargs.get("canvas", None)
        different_colors = kwargs.get("different_colors", False)

        if angle == "theta":
            q = self.theta
            P_plot = self.Ptheta / self.psi_wall
            y_label = rf"$P_\{angle}/\psi_w$"
            y_label_config = self.Config.drift_theta_ylabel_kw

        elif angle == "zeta":
            q = self.z
            P_plot = self.Pzeta
            y_label = rf"$P_\{angle}$"
            y_label_config = self.Config.drift_zeta_ylabel_kw

        # Set theta lim. Mods all thetas or zetas to 2π
        min, max = lim
        q_plot = utils.theta_plot(q, lim)

        if canvas is None:
            fig = plt.figure(figsize=(6, 4))
            ax = fig.add_subplot(111)
            canvas = (fig, ax)
        else:
            fig, ax = canvas

        scatter_kw = self.Config.drift_scatter_kw
        if different_colors:
            del scatter_kw["color"]

        ax.scatter(q_plot, P_plot, **scatter_kw, zorder=2)
        ax.set_xlabel(rf"$\{angle}$", **self.Config.drift_xlabel_kw)
        ax.set_ylabel(y_label, **y_label_config)

        # Set all xticks as multiples of π, and then re-set xlims (smart!)
        ticks = ["-2π", "-3π/2", "-π", "-π/2", "0", "π/2", "π", "3π/2", "2π"]
        ax.set_xticks(np.linspace(-2 * np.pi, 2 * np.pi, 9), ticks)
        ax.set_xlim(lim)

    def _calcW_grid(
        self,
        theta: np.array,
        psi: np.array,
        Pz: float,
        contour_Phi: bool,
        units: str,
    ):
        r"""Returns a single value or a grid of the calculated Hamiltonian.

        Only to be called internally, by ``contour_energy()``..

        Args:
            theta (np.array): The :math:`\theta` values.
            psi (np.array): The :math:`\psi` values.
            Pz (float): The :math:`P_\zeta` value.
            contour_Phi (bool): Whether or not to add the electric potential term
                :math:`e\Phi`.
            units (str): The energy units.
        """

        r = np.sqrt(2 * psi)
        B = 1 - r * np.cos(theta)
        psip = self.q.psip_of_psi(psi)

        W = (Pz + psip) ** 2 * B**2 / (2 * self.g**2 * self.mass_amu) + self.mu * B  # Without Φ

        # Add Φ if asked
        if contour_Phi:
            Phi = self.Efield.Phi_of_psi(psi)
            Phi *= self.Volts_to_NU * self.sign
            W += Phi  # all normalized

        if units == "eV":
            W *= self.NU_to_eV
        elif units == "keV":
            W *= self.NU_to_eV / 1000

        return W

    def contour_energy(
        self,
        theta_lim: list = [-np.pi, np.pi],
        psi_lim: str | list = "auto",
        plot_drift: bool = True,
        contour_Phi: bool = True,
        units: Literal["normal", "eV", "keV"] = "keV",
        levels: int = None,
        wall_shade: bool = True,
        **kwargs,
    ):
        r"""Draws a 2D contour plot of the Hamiltonian.

        Can also plot the current particle's :math:`\theta-P_\theta` drift.
        Should be False when running with multiple initial conditions.

        Args:
            theta_lim (list, optional): Plot xlim. Must be either [0,2π] or [-π,π].
                Defaults to [-π,π].
            psi_lim (list | str, optional): If a list is passed, it plots between the
                2 values relative to :math:`\psi_{wall}`. Defaults to 'auto'.
            plot_drift (bool, optional): Whether or not to plot :math:`\theta-P_\theta`
                drift on top. Defaults to True.
            contour_Phi (bool, optional): Whether or not to add the Φ term in the
                energy contour. Defaults to True.
            units (str, optional): The energy units. Must be 'normal', 'eV' or 'keV'. Defaults
                to `keV`. Defaults to "keV".
            levels (int, optional): The number of contour levels. Defaults to Config setting.
            wall_shade (bool, optional): Whether to shade the region
                :math:`\psi/\psi_{wall} > 1`. Defaults to True.
        """
        canvas = kwargs.get("canvas", None)

        if canvas is None:
            fig = plt.figure(figsize=(6, 4))
            ax = fig.add_subplot(111)
            canvas = (fig, ax)
        else:
            fig, ax = canvas

        # Set theta lim. Mods all thetas to 2π
        theta_min, theta_max = theta_lim

        if plot_drift:
            self.drift(angle="theta", theta_lim=theta_lim, canvas=canvas)

        label, E_cbar = self._cbar_label(units)

        # Set psi limits (Normalised to psi_wall)
        if type(psi_lim) is str:
            if psi_lim == "auto":
                psi_diff = self.psi.max() - self.psi.min()
                psi_mid = (self.psi.max() + self.psi.min()) / 2
                psi_lower = max(0, psi_mid - 0.6 * psi_diff)
                psi_higher = psi_mid + 0.6 * psi_diff
                psi_lim = np.array([psi_lower, psi_higher])
        else:
            psi_lim = np.array(psi_lim) * self.psi_wall
        psi_min = psi_lim[0]
        psi_max = psi_lim[1]

        # Calculate Energy values
        grid_density = self.Config.contour_grid_density
        theta, psi = np.meshgrid(
            np.linspace(theta_min, theta_max, grid_density),
            np.linspace(psi_min, psi_max, grid_density),
        )
        values = self._calcW_grid(theta, psi, self.Pz0, contour_Phi, units)
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
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$\psi/\psi_{wall}$", rotation=90)
        ticks = ["-2π", "-3π/2", "-π", "-π/2", "0", "π/2", "π", "3π/2", "2π"]
        plt.xticks(np.linspace(-2 * np.pi, 2 * np.pi, 9), ticks)
        ax.set(xlim=[theta_min, theta_max], ylim=np.array(psi_lim) / self.psi_wall)
        ax.set_facecolor("white")

        if wall_shade:  # ψ_wall boundary rectangle
            rect = Rectangle(
                (theta_lim[0], 1), 2 * np.pi, psi_max / self.psi_wall, alpha=0.2, color="k"
            )
            ax.add_patch(rect)

        if kwargs == {}:  # If called for a single particle
            label, E_cbar = self._cbar_label(units)
            cbar = fig.colorbar(C, ax=ax, fraction=0.03, pad=0.2, label=label)
            cbar.ax.plot([0, 1], [E_cbar, E_cbar], linestyle="-", zorder=3)
            return
        return C

    def _cbar_label(self, units):
        if units == "normal":
            label = "E (normalized)"
            E_cbar = self.E
        elif units == "eV":
            label = "E (eV)"
            E_cbar = self.E_eV
        elif units == "keV":
            label = "E (keV)"
            E_cbar = self.E_eV / 1000
        else:
            print('units must be either "normal", "eV" or "keV"')
            return
        return label, E_cbar

    def parabolas(self):
        """Constructs and plots the orbit type parabolas.

        Returns early if there is no Electric field.
        """
        if self.has_efield:
            print("Parabolas dont work with Efield present.")
            return
        Construct(self)

    def orbit_point(self, different_colors=False, labels=True):
        r"""Plots the particle point on the :math:`\mu-P_\zeta` (normalized) plane."""

        if self.has_efield:
            return

        orbit_point_kw = self.Config.orbit_point_kw.copy()
        if different_colors:
            del orbit_point_kw["markerfacecolor"]

        plt.plot(self.orbit_x, self.orbit_y, **orbit_point_kw)

        if labels:
            label = "  Particle " + f"({self.t_or_p[0]}-{self.l_or_c[0]})"
            plt.annotate(label, (self.orbit_x, self.orbit_y), color="b")
        plt.xlabel(r"$P_\zeta/\psi_p$")

    def _toruspoints(self, percentage: int = 100, truescale: bool = True) -> tuple:
        r"""Calculates the toroidal coordionates of the particles orbit,
        :math:`(r, \theta, \zeta)`.

        :math:`r = \sqrt{2\psi}` rather than :math:`\psi` itself is used for
        the plot, since it is a better representation of the actual orbit.

        Args:
            percentage (int, optional): The percentage of the orbit to be plotted.
                Defaults to 100.
            truescale (bool, optional): Whether or not to use the actual tokamak
                dimensions, or fit them around the orbit for better visibility.
                Defaults to True.

        Returns:
            5-tuple of np.arrays: 
                The major and minor radii of the (possibly scaled) \
                tokamak and the toroidal coordionates of the particles orbit.
                :math:`(r, \theta, \zeta)`.
        """

        if self.percentage_calculated == percentage:
            # No need to recalculate, already stored in self
            return self.Rtorus, self.atorus, self.r_torus, self.theta_torus, self.z_torus

        if percentage < 1 or percentage > 100:
            percentage = 100
            print("Invalid percentage. Plotting the whole thing.")

        points = int(np.floor(self.theta.shape[0] * percentage / 100) - 1)
        self.theta_torus = self.theta[:points]
        psi_torus = self.psi[:points]  # / self.psi_wall
        self.z_torus = self.z[:points]
        self.r_torus = np.sqrt(2 * psi_torus) * self.R  # Since r is normalized

        # Torus shape parameters
        r_span = [self.r_torus.min(), self.r_torus.max()]

        if truescale:
            self.Rtorus = self.R
            self.atorus = self.a
            # self.r_torus *= self.atorus
        else:
            self.Rtorus = (r_span[1] + r_span[0]) / 2
            self.atorus = 1.1 * self.Rtorus / 2
            self.r_torus *= 1 / 2

        self.percentage_calculated = percentage

        return self.Rtorus, self.atorus, self.r_torus, self.theta_torus, self.z_torus

    def torus2d(self, percentage: int = 100, truescale: bool = False):
        r"""Plots the poloidal and toroidal view of the orbit.

        Args:
            percentage (int, optional): 0-100: the percentage of the orbit
                to be plotted. Defaults to 100.
            truescale (bool, optional): Whether or not to construct the torus and orbit
                with the actual units of R and r. Defaults to True.
        """

        # Configure torus dimensions and orbit and store internally
        self._toruspoints(percentage=percentage, truescale=truescale)

        Rin = self.Rtorus - self.atorus
        Rout = self.Rtorus + self.atorus

        r_plot1 = self.r_torus
        r_plot2 = self.Rtorus + self.r_torus * np.cos(self.theta_torus)

        fig, ax = plt.subplots(1, 2, figsize=(8, 5), subplot_kw={"projection": "polar"})
        fig.tight_layout()

        # Torus Walls
        ax[0].scatter(
            np.linspace(0, 2 * np.pi, 1000),
            self.atorus * np.ones(1000),
            **self.Config.torus2d_wall_kw,
        )
        ax[1].scatter(
            np.linspace(0, 2 * np.pi, 1000),
            Rin * np.ones(1000),
            **self.Config.torus2d_wall_kw,
        )
        ax[1].scatter(
            np.linspace(0, 2 * np.pi, 1000),
            Rout * np.ones(1000),
            **self.Config.torus2d_wall_kw,
        )

        # Orbits
        ax[0].scatter(self.theta_torus, r_plot1, **self.Config.torus2d_orbit_kw, zorder=-1)
        ax[1].scatter(self.z_torus, r_plot2, **self.Config.torus2d_orbit_kw, zorder=-1)

        ax[0].set_ylim(bottom=0)
        ax[1].set_ylim(bottom=0)
        ax[0].grid(False)
        ax[1].grid(False)
        ax[0].set_title("Poloidal View", c="b")
        ax[1].set_title("Top-Down View", c="b")
        ax[0].set_xlabel(r"$\sqrt{2\psi} - \theta$")
        ax[1].set_xlabel(r"$\sqrt{2\psi}\cos\theta - \zeta$")
        ax[0].tick_params(labelsize=8)
        ax[1].tick_params(labelsize=8)

    def torus3d(
        self,
        percentage: int = 100,
        truescale: bool = False,
        hd: bool = False,
        bold: str = "default",
        white_background: bool = True,
    ):
        r"""Creates a 3d transparent torus and the particle's orbit.

        Args:
            percentage (int, optional): 0-100: the percentage of the orbit
                to be plotted. Defaults to 100.
            truescale (bool, optional): Whether or not to construct the torus and orbit
                with the actual units of R and r. Defaults to True.
            hd (bool, optional): High definition image (dpi = 900). Defaults to False
                (dpi = 300).
            bold (str, optional): The "boldness" level. Levels are "bold", "BOLD", or any.
                Defaults to Config settings.
            white_background (bool, optional): Whether to paint the background white or not.
                Overwrites the default plt.style(). Defaults to True.
        """

        # Configure torus dimensions and orbit and store internally
        self._toruspoints(percentage=percentage, truescale=truescale)

        custom_kw = self.Config.torus3d_orbit_kw.copy()

        if hd:
            dpi = 900
        else:
            dpi = plt.rcParams["figure.dpi"]

        if bold == "bold":
            custom_kw["alpha"] = 0.8
            custom_kw["linewidth"] *= 2
        elif bold == "BOLD":
            custom_kw["alpha"] = 1
            custom_kw["linewidth"] *= 3

        if white_background:
            bg_color = "white"
        else:
            bg_color = "k"
            custom_kw["alpha"] = 1
            custom_kw["color"] = "w"

        # Cartesian
        x = (self.Rtorus + self.r_torus * np.cos(self.theta_torus)) * np.cos(self.z_torus)
        y = (self.Rtorus + self.r_torus * np.cos(self.theta_torus)) * np.sin(self.z_torus)
        z = self.r_torus * np.sin(self.theta_torus)

        # Torus Surface
        theta_torus = np.linspace(0, 2 * np.pi, 400)
        z_torus = theta_torus
        theta_torus, z_torus = np.meshgrid(theta_torus, z_torus)
        x_torus_wall = (self.Rtorus + self.atorus * np.cos(theta_torus)) * np.cos(z_torus)
        y_torus_wall = (self.Rtorus + self.atorus * np.cos(theta_torus)) * np.sin(z_torus)
        z_torus_wall = self.atorus * np.sin(theta_torus)

        fig, ax = plt.subplots(
            dpi=dpi,
            subplot_kw={"projection": "3d"},
            **{"figsize": (10, 6), "frameon": False},
        )

        # Plot z-axis
        ax.plot([0, 0], [0, 0], [-8, 6], color="b", alpha=0.4, linewidth=0.5)
        # Plot wall surface
        ax.plot_surface(
            x_torus_wall,
            y_torus_wall,
            z_torus_wall,
            rstride=3,
            cstride=3,
            **self.Config.torus3d_wall_kw,
        )
        ax.set_axis_off()
        ax.set_facecolor(bg_color)
        ax.plot(x, y, z, **custom_kw, zorder=1)
        ax.set_box_aspect((1, 1, 0.5), zoom=1.1)
        ax.set_xlim3d(0.8 * x_torus_wall.min(), 0.8 * x_torus_wall.max())
        ax.set_ylim3d(0.8 * y_torus_wall.min(), 0.8 * y_torus_wall.max())
        ax.set_zlim3d(-3, 3)

    def fft(self):
        """Plots the timeseries and its FFT results.

        .. note:: Even though the time evolution plot only shows a few periods
            for clarity, the FFT is calculated across the full orbit.
        """

        if not self.FreqAnalysis.signal_ok:
            print("Error: cannot plot.")
            return

        x = self.FreqAnalysis.x
        t = self.FreqAnalysis.t

        X = self.FreqAnalysis.X
        omegas = self.FreqAnalysis.omegas
        base_freq = self.FreqAnalysis.omega_manual

        # Plot only a few periods:
        def plot_span():
            """Sets the timeseries plot limits"""

            tstop = 10 * 2 * np.pi / base_freq

            t_plot = t[t < tstop]
            x_plot = x[: len(t_plot)]

            return t_plot, x_plot

        def plot():
            """Does the actual plotting."""

            fig = plt.figure(figsize=(10, 5))
            fig.subplots_adjust(hspace=0.4)

            # Time evolution plot
            ax_time = fig.add_subplot(211)
            ax_time.scatter(t_plot, x_plot, **self.Config.time_scatter_kw)
            ax_time.set_xlabel(f"Time ({self.FreqAnalysis.time_unit})")
            ax_time.set_ylabel("Amplitude (rads)")

            # FFT plot
            ax_freq = fig.add_subplot(212)
            markerline, stemlines, baseline = ax_freq.stem(
                np.abs(omegas), np.abs(X), linefmt="blue"
            )
            markerline.set_markersize(4)
            stemlines.set_linewidths(0.8)
            ax_freq.set_xlabel(f"Frequency in {self.FreqAnalysis.freq_unit}.")
            ax_freq.set_ylabel("Frequency Magnitude")
            ax_freq.set_xlim([-base_freq / 5, 6 * base_freq])

        t_plot, x_plot = plot_span()
        plot()
