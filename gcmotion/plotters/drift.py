import numpy as np
import matplotlib.pyplot as plt

from gcmotion.utils._logger_setup import logger
from gcmotion.utils.pi_mod import pi_mod

from gcmotion.configuration.plot_parameters import drift as config


def drift(cwp, angle: str = "theta", lim: list = [-np.pi, np.pi], **kwargs):
    r"""Draws :math:`\theta - P_\theta` plot.

    This method is called internally by ``countour_energy()``
    as well.

    Args:
        angle (str): The angle to plot.
        lim (list, optional): Plot xlim. Must be either [0,2π] or [-π,π].
            Defaults to [-π,π].
        kwargs (list): Extra arguements if called for many particles.
    """
    logger.info(f"Plotting {angle}-P_{angle} drift...")

    # Get all needed attributes first
    q = getattr(cwp, angle)
    P_plot = getattr(cwp, "P" + angle).copy()
    psi_wall = cwp.psi_wall

    canvas = kwargs.get("canvas", None)
    different_colors = kwargs.get("different_colors", False)

    if angle == "theta":  # Normalize to psi_wall
        P_plot /= psi_wall

    # Set theta lim. Mods all thetas or zetas to 2π
    q_plot = pi_mod(q, lim)

    if canvas is None:
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        canvas = (fig, ax)
        logger.debug("\tCreating a new canvas.")
    else:  # Use external canvas
        fig, ax = canvas
        logger.debug("\tUsing existing canvas.")

    scatter_kw = config["scatter_args"]
    if different_colors and "color" in scatter_kw.keys():
        del scatter_kw["color"]

    ax.scatter(q_plot, P_plot, **scatter_kw, zorder=2)
    ax.set_xlabel(rf"$\{angle}$", fontsize=config["xfontsize"])
    ax.set_ylabel(rf"$P_\{angle}/\psi_w$", fontsize=config["yfontsize"])

    # Set all xticks as multiples of π, and then re-set xlims (smart!)
    ticks = ["-2π", "-3π/2", "-π", "-π/2", "0", "π/2", "π", "3π/2", "2π"]
    ax.set_xticks(np.linspace(-2 * np.pi, 2 * np.pi, 9), ticks)
    ax.set_xlim(lim)

    # Make interactive if single particle:
    if not kwargs:
        fig.set_tight_layout(True)
        plt.ion()
        plt.show(block=True)

    logger.info(f"{angle}-P_{angle} drift successfully plotted.")
