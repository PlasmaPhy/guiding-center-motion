""" This module initialized the Plot class, which is a component of the
composite class ``Particle``, and contains all the plotting-related methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Literal
from matplotlib.patches import Rectangle
from .parabolas import Construct
from . import utils, logger


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
        logger.info("\tCopied cwp's attributes to 'plot' object.")

    def tokamak_profile(self, zoom: list = [0, 1.1]):
        r"""Plots the electric field, potential, and q factor,
        with respect to :math:`\psi/\psi_{wall}`.

        Args:
            zoom (list, optional): zoom to specific area in the x-axis of the electric field
                and potential plots. Defaults to [0, 1.1].
        """
        logger.info("Plotting tokamak profile...")

        fig = plt.figure(dpi=200, figsize=(10, 8))
        fig.subplots_adjust(hspace=0.6)
        ax_phi = fig.add_subplot(321)
        ax_E = fig.add_subplot(322)
        ax_q1 = fig.add_subplot(323)
        ax_q2 = fig.add_subplot(324)

        psi = np.linspace(0, 1.1 * self.psi_wall, 1000)
        Er = self.Efield.Er_of_psi(psi)
        Phi = self.Efield.Phi_of_psi(psi)

        def plot_electric():
            """Plots the electric field profile in subplots 321 and 322."""
            logger.debug("\tPlotting electric field profile...")
            nonlocal psi, Er, Phi
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

            logger.debug("\t-> Electric field profile successfully plotted.")

        def plot_q():
            """Plots the q factor profile in subplots 323 and 324."""
            logger.debug("\tPlotting q factor profile...")
            nonlocal psi, Er, Phi
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

            logger.debug("\t-> Q factor profile successfully plotted.")

        def plot_magnetic():
            """Plots the magnetic field profile in a single bottom subplot."""
            logger.debug("\tPlotting electric field profile...")

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

            logger.debug("\t-> Magnetic field profile successfully plotted.")

        plot_electric()
        plot_q()
        plot_magnetic()
        logger.info("--> Tokamak profile successfully plotted.\n")

    def time_evolution(self, percentage: int = 100):
        r"""Plots the time evolution of all the dynamical variables and
        canonical momenta.

        Args:
            percentage (int, optional): The percentage of the orbit to be plotted.
                Defaults to 100.
        """

        logger.info("Plotting time evolutions...")

        if percentage < 1 or percentage > 100:
            percentage = 100
            print("Invalid percentage. Plotting the whole thing.")
            logger.warning("Invalid percentage: Plotting the whole thing...")

        points = int(np.floor(self.theta.shape[0] * percentage / 100) - 1)

        # Plotting
        fig, ax = plt.subplots(7, 1, figsize=(10, 8), sharex=True, dpi=300)
        fig.tight_layout()
        ax[0].set_title("Time evolution of dynamical variables", c="b")
        ax[5].set_title("Time evolution of canonical momenta", c="b")

        ax[0].scatter(self.t_eval[:points], self.theta[:points], **self.configs["time_plots_kw"])
        ax[1].scatter(self.t_eval[:points], self.zeta[:points], **self.configs["time_plots_kw"])
        ax[2].scatter(self.t_eval[:points], self.psi[:points], **self.configs["time_plots_kw"])
        ax[3].scatter(self.t_eval[:points], self.psip[:points], **self.configs["time_plots_kw"])
        ax[4].scatter(self.t_eval[:points], self.rho[:points], **self.configs["time_plots_kw"])
        ax[5].scatter(self.t_eval[:points], self.Ptheta[:points], **self.configs["time_plots_kw"])
        ax[6].scatter(self.t_eval[:points], self.Pzeta[:points], **self.configs["time_plots_kw"])

        ax[0].set_ylabel(r"$\theta(t)$", **self.configs["time_plots_ylabel_kw"])
        ax[1].set_ylabel(r"$\zeta(t)$", **self.configs["time_plots_ylabel_kw"])
        ax[2].set_ylabel(r"$\psi(t)$", **self.configs["time_plots_ylabel_kw"])
        ax[3].set_ylabel(r"$\psi_p(t)$", **self.configs["time_plots_ylabel_kw"])
        ax[4].set_ylabel(r"$\rho(t)$", **self.configs["time_plots_ylabel_kw"])
        ax[5].set_ylabel(r"$P_\theta(t)$", **self.configs["time_plots_ylabel_kw"])
        ax[6].set_ylabel(r"$P_\zeta(t)$", **self.configs["time_plots_ylabel_kw"])
        ax[6].set_ylim([-self.psip_wall, self.psip_wall])

        plt.xlabel("$t$")

        logger.info("--> Time evolutions successfully plotted.\n")

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

        ax[0].scatter(theta_plot, self.Ptheta, **self.configs["drift_scatter_kw"])
        ax[1].scatter(self.zeta, self.Pzeta, **self.configs["drift_scatter_kw"])

        ax[0].set_xlabel(r"$\theta$", fontsize=self.configs["drift_plots_xlabel_fontsize"])
        ax[1].set_xlabel(r"$\zeta$", fontsize=self.configs["drift_plots_xlabel_fontsize"])

        ax[0].set_ylabel(r"$P_\theta$", fontsize=self.configs["drift_plots_ylabel_fontsize"])
        ax[1].set_ylabel(r"$P_ζ$", fontsize=self.configs["drift_plots_ylabel_fontsize"])

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
            fontsize = self.configs["drift_plots_ylabel_fontsize"]

        elif angle == "zeta":
            q = self.zeta
            P_plot = self.Pzeta
            y_label = rf"$P_\{angle}$"
            fontsize = self.configs["drift_plots_ylabel_fontsize"]

        # Set theta lim. Mods all thetas or zetas to 2π
        min, max = lim
        q_plot = utils.theta_plot(q, lim)

        if canvas is None:
            fig = plt.figure(figsize=(6, 4))
            ax = fig.add_subplot(111)
            canvas = (fig, ax)
        else:
            fig, ax = canvas

        scatter_kw = self.configs["drift_scatter_kw"]
        if different_colors:
            del scatter_kw["color"]

        ax.scatter(q_plot, P_plot, **scatter_kw, zorder=2)
        ax.set_xlabel(rf"$\{angle}$", fontsize=self.configs["drift_plots_xlabel_fontsize"])
        ax.set_ylabel(y_label, fontsize=fontsize)

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
        if contour_Phi:
            logger.debug("\tAdding Φ term to the contour.")
        logger.debug(f"\tCalculating energy values in a {theta.shape} grid.")

        r = np.sqrt(2 * psi)
        B = self.Bfield.B(r, theta)
        psip = self.q.psip_of_psi(psi)

        W = (Pz + psip) ** 2 * B**2 / (
            2 * self.Bfield.g**2 * self.mass_amu
        ) + self.mu * B  # Without Φ

        # Add Φ if asked
        if contour_Phi:
            Phi = self.Efield.Phi_of_psi(psi)
            Phi *= self.Volts_to_NU * self.sign
            W += Phi  # all normalized

        if units == "eV":
            W *= self.NU_to_eV
            logger.debug("\tPlotting energy levels in [eV]")
        elif units == "keV":
            W *= self.NU_to_eV / 1000
            logger.debug("\tPlotting energy levels in [keV]")
        else:
            logger.debug("\tPlotting energy levels in [NU]")

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
        logger.info("Plotting energy contour:")
        canvas = kwargs.get("canvas", None)

        if canvas is None:
            fig = plt.figure(figsize=(6, 4))
            ax = fig.add_subplot(111)
            canvas = (fig, ax)
            logger.debug("\tCreating a new canvas.")
        else:
            fig, ax = canvas
            logger.debug("\tUsing existing canvas.")

        # Set theta lim. Mods all thetas to 2π
        theta_min, theta_max = theta_lim

        if plot_drift:
            self.drift(angle="theta", theta_lim=theta_lim, canvas=canvas)
            logger.debug("\tPlotting particle's Pθ drift.")

        E_cbar = self._cbar_energy(units)

        # Set psi limits (Normalised to psi_wall)
        if type(psi_lim) is str:
            if psi_lim == "auto":
                psi_diff = self.psi.max() - self.psi.min()
                psi_mid = (self.psi.max() + self.psi.min()) / 2
                psi_lower = max(0, psi_mid - 0.6 * psi_diff)
                psi_higher = psi_mid + 0.6 * psi_diff
                psi_lim = np.array([psi_lower, psi_higher])
                logger.debug("\tUsing automatic ψ limits.")
        else:
            psi_lim = np.array(psi_lim) * self.psi_wall
            logger.debug("\tUsing user-defined ψ limits.")
        psi_min = psi_lim[0]
        psi_max = psi_lim[1]

        # Calculate Energy values
        grid_density = self.configs["contour_grid_density"]
        theta, psi = np.meshgrid(
            np.linspace(theta_min, theta_max, grid_density),
            np.linspace(psi_min, psi_max, grid_density),
        )
        values = self._calcW_grid(theta, psi, self.Pz0, contour_Phi, units)
        span = np.array([values.min(), values.max()])
        logger.debug(f"\tEnergy values span from {span[0]:.4g}{units} to {span[1]:.4g}{units}.")

        # Create Figure
        if levels is None:  # If non is given
            levels = self.configs["contour_levels_default"]
            logger.debug("\tUsing default number of levels.")
        else:
            logger.debug(f"\tOverwritting default levels number to {levels}")
        contour_kw = {
            "vmin": span[0],
            "vmax": span[1],
            "levels": levels,
            "cmap": self.configs["contour_cmap"],
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
            logger.debug("\tAdding wall shade.")

        if not kwargs:  # If called for a single particle
            cbar = fig.colorbar(C, ax=ax, fraction=0.03, pad=0.2, label=f"E[{units}]")
            cbar_kw = {"linestyle": "-", "zorder": 3}
            E_cbar = self._cbar_energy(units)
            cbar.ax.plot([0, 1], [E_cbar, E_cbar], **cbar_kw)
            logger.debug(f"\tSingle particle call. Adding energy label at {E_cbar:.4g}{units}")

        if not kwargs:  # If called for a single particle
            logger.info("--> Energy contour successfully plotted (returned null)\n")
        elif kwargs:  # If called for a collection
            logger.info("--> Energy contour successfully plotted (returned contour object)\n")
            return C

    def _cbar_energy(self, units):
        """Creates a colorbar label of the particle's energy.

        Args:
            units (str): The energy units.

        Returns:
            float: The energy value.
        """
        if units == "normal":
            E_cbar = self.E
        elif units == "eV":
            E_cbar = self.E_eV
        elif units == "keV":
            E_cbar = self.E_eV / 1000
        else:
            print('units must be either "normal", "eV" or "keV"')
            return

        return E_cbar

    def parabolas(self):
        """Constructs and plots the orbit type parabolas.

        Returns early if there is no Electric field.
        """
        logger.info("Plotting orbit type Parabolas:")
        if self.has_efield or not self.Bfield.is_lar:
            print("Parabolas dont work with Efield present.")
            logger.info(
                "\tElectric field is present, or Magnetic field is not LAR. Orbit type parabolas do not work."
            )
            return
        logger.debug("Calling 'Construct' class")
        Construct(self)
        logger.info("--> Parabolas and Boundary plotted successfully.\n")

    def orbit_point(self, different_colors=False, labels=True):
        r"""Plots the particle point on the :math:`\mu-P_\zeta` (normalized) plane."""
        logger.info("Plotting orbit type point on parabolas plot...")

        orbit_point_kw = self.configs["orbit_point_kw"]
        if different_colors:
            logger.debug("\tUsing different colors for each particle.")
            del orbit_point_kw["markerfacecolor"]

        plt.scatter(self.orbit_x, self.orbit_y, **orbit_point_kw)

        if labels:
            label = "  Particle " + f"({self.t_or_p[0]}-{self.l_or_c[0]})"
            plt.annotate(label, (self.orbit_x, self.orbit_y), color="b")
            logger.debug("\tPlotting particle's labels.")

        plt.xlabel(r"$P_\zeta/\psi_p$")
        logger.info("--> Plotted orbit type point successfully.\n")

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
        logger.info("Calculating torus plotting points...")

        if self.percentage_calculated == percentage and self.truescale_calculated == truescale:
            # No need to recalculate, already stored in self
            logger.info(
                "--> Points already calculated for that percentage and scale. Returning those..."
            )
            logger.info(
                f"--> Calculation successful. Now stored: percentage = {self.percentage_calculated}, truescale = {self.truescale_calculated}."
            )
            return self.Rtorus, self.atorus, self.r_torus, self.theta_torus, self.z_torus

        if percentage < 1 or percentage > 100:
            percentage = 100
            print("Invalid percentage. Plotting the whole thing.")
            logger.warning("Invalid percentage: Plotting the whole thing...")

        points = int(np.floor(self.theta.shape[0] * percentage / 100) - 1)
        self.theta_torus = self.theta[:points]
        psi_torus = self.psi[:points]  # / self.psi_wall
        self.z_torus = self.zeta[:points]
        self.r_torus = np.sqrt(2 * psi_torus) * self.R  # Since r is normalized

        # Torus shape parameters
        r_span = [self.r_torus.min(), self.r_torus.max()]
        logger.debug(f"\tr-span calculated:[{r_span[0]:.4g}, {r_span[1]:.4g}]m, with a={self.a}m.")

        if truescale:
            self.Rtorus = self.R
            self.atorus = self.a
            logger.debug("\tPlotting the orbit in True scale.")
        else:
            self.Rtorus = (r_span[1] + r_span[0]) / 2
            self.atorus = 1.1 * self.Rtorus / 2
            self.r_torus *= 1 / 2
            logger.warning("Plotting the zoomed in obrit.")

        self.percentage_calculated = percentage
        self.truescale_calculated = truescale
        logger.info(
            f"--> Calculation successful. Now stored: percentage = {self.percentage_calculated}, truescale = {self.truescale_calculated}."
        )

        return self.Rtorus, self.atorus, self.r_torus, self.theta_torus, self.z_torus

    def torus2d(self, percentage: int = 100, truescale: bool = False):
        r"""Plots the poloidal and toroidal view of the orbit.

        Args:
            percentage (int, optional): 0-100: the percentage of the orbit
                to be plotted. Defaults to 100.
            truescale (bool, optional): Whether or not to construct the torus and orbit
                with the actual units of R and r. Defaults to True.
        """
        logger.info("Plotting 2D torus sections...")
        # Configure torus dimensions and orbit and store internally
        self._toruspoints(percentage=percentage, truescale=truescale)

        Rin = self.Rtorus - self.atorus
        Rout = self.Rtorus + self.atorus
        logger.debug(f"Calculated Rin = {Rin:.4g}, Rout = {Rout:.4g}.")

        r_plot1 = self.r_torus
        r_plot2 = self.Rtorus + self.r_torus * np.cos(self.theta_torus)

        fig, ax = plt.subplots(1, 2, figsize=(8, 5), subplot_kw={"projection": "polar"})
        fig.tight_layout()

        # Torus Walls
        ax[0].scatter(
            np.linspace(0, 2 * np.pi, 1000),
            self.atorus * np.ones(1000),
            **self.configs["torus2d_wall_kw"],
        )
        ax[1].scatter(
            np.linspace(0, 2 * np.pi, 1000),
            Rin * np.ones(1000),
            **self.configs["torus2d_wall_kw"],
        )
        ax[1].scatter(
            np.linspace(0, 2 * np.pi, 1000),
            Rout * np.ones(1000),
            **self.configs["torus2d_wall_kw"],
        )

        # Orbits
        ax[0].scatter(self.theta_torus, r_plot1, **self.configs["torus2d_orbit_kw"], zorder=-1)
        ax[1].scatter(self.z_torus, r_plot2, **self.configs["torus2d_orbit_kw"], zorder=-1)

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

        logger.info("--> 2D torus sections plotted successfully.\n")

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
        logger.info("Plotting 3D torus...")
        # Configure torus dimensions and orbit and store internally
        self._toruspoints(percentage=percentage, truescale=truescale)

        custom_kw = self.configs["torus3d_orbit_kw"]

        if hd:
            dpi = 900
            logger.debug(f"\tPlotting image in HD ({int(dpi)}dpi).")
        else:
            dpi = plt.rcParams["figure.dpi"]
            logger.debug(f"\tPlotting image in default definition ({int(dpi)}dpi).")

        if bold == "bold":
            custom_kw["alpha"] = 0.8
            custom_kw["linewidth"] *= 2
        elif bold == "BOLD":
            custom_kw["alpha"] = 1
            custom_kw["linewidth"] *= 3
        logger.debug(
            f"\tOrbit plot size: {bold} (linewidth = {custom_kw["linewidth"]}, alpha = {custom_kw["alpha"]})."
        )

        if white_background:
            bg_color = "white"
        else:
            bg_color = "k"
            custom_kw["alpha"] = 1
            custom_kw["color"] = "w"
        logger.debug(f"\tUsing white background: {white_background}")

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
            **self.configs["torus3d_wall_kw"],
        )
        ax.set_axis_off()
        ax.set_facecolor(bg_color)
        ax.plot(x, y, z, **custom_kw, zorder=1)
        ax.set_box_aspect((1, 1, 0.5), zoom=1.1)
        ax.set_xlim3d(0.8 * x_torus_wall.min(), 0.8 * x_torus_wall.max())
        ax.set_ylim3d(0.8 * y_torus_wall.min(), 0.8 * y_torus_wall.max())
        ax.set_zlim3d(-3, 3)

        logger.info("--> 3D torus plotted successfullly.\n")

    def _fft(self, obj):
        """Plots the calculated DFT results

        Args:
            obj (FreqAnalysis): object containing the frequency analysis results
        """

        signal = obj.signal
        t_signal = obj.t_signal

        X = obj.X
        omegas = obj.omegas

        base_freq = obj.base_freq

        def plot():
            """Does the actual plotting."""

            fig = plt.figure(figsize=(10, 5))
            fig.subplots_adjust(hspace=0.4)

            # Time evolution plot
            ax_time = fig.add_subplot(211)
            ax_time.scatter(t_signal, signal, **self.configs["time_plots_kw"])
            ax_time.set_xlabel("Time [s]")
            ax_time.set_ylabel("Amplitude (rads)")

            # FFT plot
            ax_freq = fig.add_subplot(212)
            markerline, stemlines, baseline = ax_freq.stem(omegas, X, linefmt="blue")
            markerline.set_markersize(4)
            stemlines.set_linewidths(0.8)
            ax_freq.set_xlabel("Frequency in Hz.")
            ax_freq.set_ylabel("Frequency Magnitude")
            # ax_freq.semilogy()
            # ax_freq.set_ylim(bottom=1)
            # ax_freq.set_xlim([-base_freq / 5, 6 * base_freq])

            return fig, ax_time, ax_freq

        def found_points():
            nonlocal fig, ax_time, ax_freq

            # Found frequencies points
            height = X.max()
            ax_freq.set_ylim(top=1.3 * height)
            # Zeroth frequency
            if obj.angle == "zeta":
                ax_freq.scatter(omegas[obj.fft_peak_index[0]], 0, c="r", zorder=5)
                ax_freq.annotate(
                    f"{omegas[obj.fft_peak_index[0]]:.4g}Hz",
                    (omegas[obj.fft_peak_index[0]], height / 3),
                    color="r",
                    zorder=4,
                )

            # actual harmonics
            for i in obj.found_indeces:
                ax_freq.scatter(omegas[i], 0, c="r", zorder=3)
                ax_freq.annotate(f"{omegas[i]:.4g}Hz", (omegas[i], height / 3), color="r", zorder=4)

            # left-and-right peaks
            for i in obj.fft_peak_index:
                ax_freq.scatter(omegas[i], 0, c="k", zorder=2)

            # Event locator results
            ax_freq.plot(
                [obj.theta_freq_event, obj.theta_freq_event], [1.2 * height, 0], c="g", zorder=-1
            )  # θ
            ax_freq.annotate(
                f"event θ:\n{obj.theta_freq_event:.4g}Hz",
                (obj.theta_freq_event, height),
                color="g",
                zorder=4,
            )
            ax_freq.plot(
                [obj.zeta_0freq_event, obj.zeta_0freq_event], [1.2 * height, 0], c="g", zorder=-1
            )  # ζ0
            ax_freq.annotate(
                f"event z0:\n{obj.zeta_0freq_event:.4g}Hz",
                (obj.zeta_0freq_event, height),
                color="g",
                zorder=4,
            )
            ax_freq.plot(
                [obj.zeta_freq_event, obj.zeta_freq_event], [1.2 * height, 0], c="g", zorder=-1
            )  # ζ
            ax_freq.annotate(
                f"event z:\n{obj.zeta_freq_event:.4g}Hz",
                (obj.zeta_freq_event, 0.7 * height),
                color="g",
                zorder=4,
            )

        def legend():
            nonlocal fig, ax_time, ax_freq

            label_event = "Event locator results"
            label_fft_peaks = "Peaks calculated from FFT"
            label_fft_freqs = "Frequencies calculated from FFT"
            colors = ["g", "k", "r"]
            ax_freq.legend(
                labels=[label_event, label_fft_peaks, label_fft_freqs],
                labelcolor=colors,
                markerscale=0,
            )

        fig, ax_time, ax_freq = plot()
        found_points()
        legend()
        fig.tight_layout()
