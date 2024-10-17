import numpy as np
import matplotlib.pyplot as plt

from gcmotion.utils._logger_setup import logger
from gcmotion.utils.pi_mod import pi_mod

from gcmotion.configuration.plot_parameters import drift as config


def drifts(cwp, theta_lim: list = [-np.pi, np.pi], **kwargs):
    r"""Draws 2 plots: 1] :math:`\theta-P_\theta`
    and 2] :math:`\zeta-P_\zeta`.

    Args:
        theta_lim (list, optional): Plot xlim. Must be either [0,2π] or [-π,π].
            Defaults to [-π,π].
        kwargs (list): Extra arguements if called for many particles.
    """
    logger.info("Plotting θ-Pθ and ζ-Pζ drifts...")

    # Get all needed attributes first
    psip_wall = cwp.psip_wall
    theta = cwp.theta
    Ptheta = cwp.Ptheta
    zeta = cwp.zeta
    Pzeta = cwp.Pzeta

    # Set theta lim. Mods all thetas to 2π
    theta_min, theta_max = theta_lim
    theta_plot = pi_mod(theta, theta_lim)

    canvas = kwargs.get("canvas", None)
    different_colors = kwargs.get("different_colors", False)

    if canvas is None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        fig.tight_layout()
        canvas = (fig, ax)
        logger.debug("\tCreating a new canvas.")
    else:  # Use external canvas
        fig, ax = canvas
        logger.debug("\tUsing existing canvas.")

    fig.suptitle(r"Drift orbits of $P_\theta - \theta$ and $P_\zeta - \zeta$")

    scatter_kw = config["scatter_args"]
    if different_colors and "color" in scatter_kw.keys():
        del scatter_kw["color"]

    ax[0].scatter(theta_plot, Ptheta, **config["scatter_args"])
    ax[1].scatter(zeta, Pzeta, **config["scatter_args"])

    ax[0].set_xlabel(r"$\theta$", fontsize=config["xfontsize"])
    ax[1].set_xlabel(r"$\zeta$", fontsize=config["xfontsize"])

    ax[0].set_ylabel(r"$P_\theta$", fontsize=config["yfontsize"])
    ax[1].set_ylabel(r"$P_ζ$", fontsize=config["yfontsize"])

    ax[1].set_ylim([-psip_wall, psip_wall])

    # Set all xticks as multiples of π, and then re-set xlims (smart!)
    ticks = ["-2π", "-3π/2", "-π", "-π/2", "0", "π/2", "π", "3π/2", "2π"]
    ax[0].set_xticks(np.linspace(-2 * np.pi, 2 * np.pi, 9), ticks)
    ax[0].set_xlim(theta_lim)

    # Make interactive if single particle:
    if not kwargs:
        fig.set_tight_layout(True)
        plt.ion()
        plt.show(block=True)

    logger.info("θ-Pθ and ζ-Pζ drifts successfully plotted.")
