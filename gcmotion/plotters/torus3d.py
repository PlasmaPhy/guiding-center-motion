import numpy as np
import matplotlib.pyplot as plt

from gcmotion.utils._logger_setup import logger

from gcmotion.configuration.plot_parameters import torus3d as config

from gcmotion.utils.canonical_to_toroidal import canonical_to_toroidal


def torus3d(
    cwp,
    percentage: int = 100,
    truescale: bool = True,
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
    Rtorus, atorus, r_torus, theta_torus, z_torus = canonical_to_toroidal(
        cwp, percentage=percentage, truescale=truescale
    )
    custom_kw = config["torus3d_orbit_kw"]

    if hd:
        dpi = 900
        logger.debug(f"\tPlotting image in HD ({int(dpi)}dpi).")
    else:
        dpi = 300
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
    x = (Rtorus + r_torus * np.cos(theta_torus)) * np.cos(z_torus)
    y = (Rtorus + r_torus * np.cos(theta_torus)) * np.sin(z_torus)
    z = r_torus * np.sin(theta_torus)

    # Torus Surface
    theta_values = z_values = np.linspace(0, 2 * np.pi, config["wall_points"])
    theta_grid, z_grid = np.meshgrid(theta_values, z_values)
    x_torus_wall = (Rtorus + atorus * np.cos(theta_grid)) * np.cos(z_grid)
    y_torus_wall = (Rtorus + atorus * np.cos(theta_grid)) * np.sin(z_grid)
    z_torus_wall = atorus * np.sin(theta_grid)

    fig, ax = plt.subplots(
        1,
        subplot_kw={"projection": "3d"},
        **{"figsize": (10, 10), "frameon": False},
    )

    # Plot z-axis
    ax.plot([0, 0], [0, 0], [-8, 6], color="b", alpha=0.4, linewidth=0.5)
    # Plot wall surface
    ax.plot_surface(
        x_torus_wall,
        y_torus_wall,
        z_torus_wall,
        rstride=config["rstride"],
        cstride=config["cstride"],
        **config["torus3d_wall_kw"],
    )

    ax.set_axis_off()
    ax.set_facecolor(bg_color)
    ax.plot(x, y, z, **custom_kw, zorder=1)
    ax.set_box_aspect((1, 1, 1), zoom=1.6)
    ax.set_xlim3d(0.8 * x_torus_wall.min(), 0.8 * x_torus_wall.max())
    ax.set_ylim3d(0.8 * y_torus_wall.min(), 0.8 * y_torus_wall.max())
    ax.set_zlim3d(-3, 3)

    ax.patch.set_edgecolor("black")

    ax.patch.set_linewidth(1)

    fig.set_tight_layout(True)
    plt.autoscale(tight=True)
    plt.ion()
    plt.show(block=True)

    logger.info("--> 3D torus plotted successfullly.\n")
