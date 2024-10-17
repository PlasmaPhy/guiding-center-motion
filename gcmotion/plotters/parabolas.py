import matplotlib.pyplot as plt

from gcmotion.utils._logger_setup import logger

from gcmotion.classes.parabolas import Construct

from gcmotion.configuration.plot_parameters import orbit_point as config


def parabolas(cwp, plot_point: bool = True):
    """Constructs and plots the orbit type parabolas.

    Returns early if there is no Electric field.
    """
    logger.info("Plotting orbit type Parabolas:")

    if cwp.has_efield or not cwp.Bfield.is_lar:
        string = "Electric field is present, or Magnetic field is not LAR. Orbit type parabolas do not work."
        print(string)
        logger.info("\t" + string)
        return
    logger.debug("Calling 'Construct' class")
    obj = Construct(cwp)
    canvas = obj.get_canvas()
    logger.info("--> Parabolas and Boundary plotted successfully.\n")

    fig, ax = canvas

    if plot_point:
        orbit_point(cwp, canvas)

    plt.ion()
    plt.show(block=True)


# ----------------------------------------------------------------


def orbit_point(cwp, canvas=None, **kwargs):
    r"""Plots the particle point on the :math:`\mu-P_\zeta` (normalized) plane."""
    logger.info("Plotting orbit type point on parabolas plot...")

    # Get all needed attributes first
    orbit_x = cwp.orbit_x
    orbit_y = cwp.orbit_y
    t_or_p = cwp.t_or_p
    l_or_c = cwp.l_or_c

    if canvas is None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        fig.tight_layout()
        canvas = (fig, ax)
        logger.debug("\tCreating a new canvas.")
    else:  # Use external canvas
        fig, ax = canvas
        logger.debug("\tUsing existing canvas.")

    different_colors = kwargs.get("different_colors", False)
    labels = kwargs.get("labels", True)

    fig, ax = canvas

    if cwp.has_efield or not cwp.Bfield.is_lar:
        string = "Electric field is present, or Magnetic field is not LAR. Orbit type point cannote be plotted."
        print(string)
        logger.info("\t" + string)
        return

    orbit_point_kw = config["orbit_point_kw"]
    if different_colors and "markerfacecolor" in orbit_point_kw.keys():
        logger.debug("\tUsing different colors for each particle.")
        del orbit_point_kw["markerfacecolor"]

    ax.scatter(orbit_x, orbit_y, **orbit_point_kw)

    if labels:
        label = "  Particle " + f"({t_or_p[0]}-{l_or_c[0]})"
        ax.annotate(label, (orbit_x, orbit_y), color="b")
        logger.debug("\tPlotting particle's labels.")

    ax.set_xlabel(r"$P_\zeta/\psi_p$")
    logger.info("--> Plotted orbit type point successfully.\n")
