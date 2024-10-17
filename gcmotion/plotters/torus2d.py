import numpy as np
import matplotlib.pyplot as plt

from gcmotion.utils._logger_setup import logger

from gcmotion.configuration.plot_parameters import torus2d as config

from gcmotion.utils.canonical_to_toroidal import canonical_to_toroidal


def torus2d(cwp, percentage: int = 100, truescale: bool = True):
    r"""Plots the poloidal and toroidal view of the orbit.

    Args:
        percentage (int, optional): 0-100: the percentage of the orbit
            to be plotted. Defaults to 100.
        truescale (bool, optional): Whether or not to construct the torus and orbit
            with the actual units of R and r. Defaults to True.
    """
    logger.info("Plotting 2D torus sections...")
    # Configure torus dimensions and orbit and store internally
    Rtorus, atorus, r_torus, theta_torus, z_torus = canonical_to_toroidal(
        cwp, percentage=percentage, truescale=truescale
    )

    Rin = Rtorus - atorus
    Rout = Rtorus + atorus
    logger.debug(f"Calculated Rin = {Rin:.4g}, Rout = {Rout:.4g}.")

    r_plot1 = r_torus
    r_plot2 = Rtorus + r_torus * np.cos(theta_torus)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6), subplot_kw={"projection": "polar"})

    wall_points = config["wall_points"]

    # Torus Walls
    ax[0].scatter(
        np.linspace(0, 2 * np.pi, wall_points),
        atorus * np.ones(wall_points),
        **config["torus2d_wall_kw"],
    )
    ax[1].scatter(
        np.linspace(0, 2 * np.pi, wall_points),
        Rin * np.ones(wall_points),
        **config["torus2d_wall_kw"],
    )
    ax[1].scatter(
        np.linspace(0, 2 * np.pi, wall_points),
        Rout * np.ones(wall_points),
        **config["torus2d_wall_kw"],
    )

    # Orbits
    ax[0].scatter(theta_torus, r_plot1, **config["torus2d_orbit_kw"], zorder=-1)
    ax[1].scatter(z_torus, r_plot2, **config["torus2d_orbit_kw"], zorder=-1)

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
    ax[0].set_axis_off()
    ax[1].set_axis_off()

    logger.info("--> 2D torus sections plotted successfully.\n")

    fig.set_tight_layout(True)
    plt.ion()
    plt.show(block=True)
