import numpy as np
import matplotlib.pyplot as plt
from .parabolas import Construct
from . import utils
from typing import Literal
from . import config


class Plots:
    r"""Component of class ``Collection``. Contains all the plotting-related methods."""

    def __init__(self, collection):
        r"""Copies attributes from *collection* to self.

        The instance itself is automatically initialized internally by the ``Collection``
        class, and only its methods are to be called by the user, as
        ``collection.plot.<method()>``.

        Args:
            collection (Particle): The Particle collection
        """

        self.__dict__ = dict(collection.__dict__)

        # Grab configuration
        self.configs = config.configs

    def _check_multiples(self, allowed: list) -> bool:
        """Checks if the given parameters are static or vary from particle to particle.

        Since all the plots require some parameters to be the same and allowing only
        certain parameters to vary from particle to particle, this check is important.
        Otherwise the resulting plots are nonsense.

        Args:
            allowed (list): The parameters which are alloed to vary.

        Returns:
            bool: Truth value depending on the result of the check.
        """
        for key in allowed:
            expr = "self.multiple_" + key + " is True"
            if eval(expr):
                print("Error")
                return False
        return True

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
        r"""Creates contour plot with the particles' drifts on top.

        Uses mostly the already existing plotting methods from ``Particle``

        The particles **must** have the same R, a, qfactor, efield, bfield,
        mu, Pz0 and be the same species.

        Args:
            theta_lim (list, optional): Plot xlim. Must be either [0,2π] or [-π,π]. Defaults to [-np.pi, np.pi].
            psi_lim (str | list, optional): If a list is passed, it plots between the
                2 values relative to :math:`\psi_{wall}`. Defaults to "auto".
            plot_drift (bool, optional): Whether or not to plot
                :math:`\theta - P_\theta` drift on top.. Defaults to True.
            contour_Phi (bool, optional): Whether or not to add the Φ term in the
                energy contour.. Defaults to True.
            units (str, optional): The energy units. Must be 'normal', 'eV' or 'keV'. Defaults
                to `keV`. Defaults to "keV".
            levels (int, optional): The number of contour levels.. Defaults to None.
            wall_shade (bool, optional): Whether to shade the region
                :math:`\psi/ \psi_{wall} > 1`. Defaults to True.
        """

        different_colors = kwargs.get("different_colors", False)
        plot_initial = kwargs.get("plot_initial", False)

        def params_ok() -> bool:
            """checks for the validity of the parameters

            Returns:
                bool: The check result
            """
            must_be_the_same = ["R", "a", "q", "Bfield", "Efield", "species", "mu", "Pz0"]
            can_be_different = [key for key in self.params.keys() if key not in must_be_the_same]

            if not self._check_multiples(must_be_the_same):
                print(f"Only the variables {can_be_different} may vary from particle to particle.")
                return False
            return True

        def plot():
            """Does the actual plotting"""
            # Setup canvas (must be passeed to all methods)
            fig = plt.figure(figsize=(6, 4), dpi=300)
            ax = fig.add_subplot(111)

            # Plot drifts:
            for p in self.particles:
                p.plot.drift(
                    angle="theta",
                    lim=theta_lim,
                    canvas=(fig, ax),
                    different_colors=different_colors,
                )

            # Plot starting points
            if plot_initial:
                for p in self.particles:
                    ax.scatter(p.theta0, p.psi0 / p.psi_wall, s=10, c="k", zorder=4)

            # Set psi limits (Normalised to psi_wall)
            nonlocal psi_lim
            if type(psi_lim) is str:
                if psi_lim == "auto":
                    psi_lim = list(ax.get_ylim())
                    psi_lim[0] = max(psi_lim[0], 0)
                    psi_lim = tuple(psi_lim)
            else:
                psi_lim = np.array(psi_lim)

            # Just use the already existing method in one of the particles, doesnt matter
            C = p.plot.contour_energy(
                theta_lim=theta_lim,
                psi_lim=psi_lim,
                plot_drift=False,
                different_colors=False,
                contour_Phi=contour_Phi,
                units=units,
                levels=levels,
                wall_shade=wall_shade,
                canvas=(fig, ax),
            )

            # Plot Energy labels on colorbar
            label, _ = p.plot._cbar_energy(units)
            cbar_kw = {"linestyle": "-", "zorder": 3}
            if not different_colors:
                cbar_kw["color"] = self.configs["drift_scatter_kw"]["color"]
            cbar = fig.colorbar(C, ax=ax, fraction=0.03, pad=0.2, label=label)
            for p in self.particles:
                E_cbar = p.plot._cbar_energy(units)
                cbar.ax.plot([0, 1], [E_cbar, E_cbar], **cbar_kw)

        if params_ok():
            plot()
        return

    def parabolas(self, different_colors=True, labels=False):
        """Plots the orbit type parabolas and the particles' orbit type points.

        .. caution:
            The particles **must** have the same R, a, qfactor, efield, bfield,
            mu, Pz0 and be the same species.

        Args:
            different_colors (bool, optional): Whether or not to use different
                different color for each drift and colorbar label. Defaults to True.
            labels (bool, optional): Whether or not to print the particles'
                labels above their orbit type points. Defaults to False.
        """

        def params_ok() -> bool:
            """checks for the validity of the parameters

            Returns:
                bool: The check result
            """

            must_be_the_same = ["R", "a", "q", "Bfield", "Efield", "species", "mu"]
            can_be_different = [key for key in self.params.keys() if key not in must_be_the_same]

            if not self._check_multiples(must_be_the_same):
                print(f"Only the variables {can_be_different} may vary from particle to particle.")
                return False

            if self.particles[0].has_efield:
                print("Parabolas dont work with Efield present.")
                return False

            if not self.params["Bfield"].is_lar:
                print("Orbit type calculations apply only in LAR.")
                return False
            return True

        def plot():
            """Does the actual plotting."""
            Construct(self.particles[0], limit_axis=False)

            for p in self.particles:
                p.plot.orbit_point(different_colors=different_colors, labels=labels)

        if params_ok():
            plot()
        return

    def poincare(
        self, angle: Literal["zeta", "theta"] = "theta", lim: list = [-np.pi, np.pi], **kwargs
    ):

        different_colors = kwargs.get("different_colors", False)
        plot_initial = kwargs.get("plot_initial", False)

        def params_ok() -> bool:
            """checks for the validity of the parameters

            Returns:
                bool: The check result
            """

            if angle == "theta":
                same_initial_momenta = "Pz0"
            elif angle == "zeta":
                same_initial_momenta = "psi0"

            must_be_the_same = [
                "R",
                "a",
                "q",
                "Bfield",
                "Efield",
                "species",
                "mu",
                same_initial_momenta,
            ]
            can_be_different = [key for key in self.params.keys() if key not in must_be_the_same]

            if not self._check_multiples(must_be_the_same):
                print(f"Only the variables {can_be_different} may vary from particle to particle.")
                return False
            return True

        def plot():
            """Does the actual plotting"""
            # Setup canvas (must be passeed to all methods)
            fig = plt.figure(figsize=(6, 4), dpi=300)
            ax = fig.add_subplot(111)

            # Plot drifts:
            for p in self.particles:
                p.plot.drift(angle, lim, canvas=(fig, ax), different_colors=different_colors)

            # Plot starting points
            if plot_initial and angle == "theta":
                for p in self.particles:
                    ax.scatter(p.theta0, p.psi0 / p.psi_wall, s=10, c="k", zorder=4)
            elif plot_initial and angle == "zeta":
                for p in self.particles:
                    ax.scatter(p.z0, p.Pz0, s=10, c="k", zorder=4)

        if params_ok():
            plot()
        return
