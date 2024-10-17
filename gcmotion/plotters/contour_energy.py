import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from gcmotion.utils._logger_setup import logger

from gcmotion.plotters.drift import drift

from gcmotion.configuration.plot_parameters import contour_energy as config


def contour_energy(
    cwp,
    theta_lim: list = [-np.pi, np.pi],
    psi_lim: str | list = "auto",
    plot_drift: bool = True,
    contour_Phi: bool = True,
    units: str = "keV",
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

    # Get all needed attributes first
    Pzeta0 = cwp.Pzeta0
    psi_wall = cwp.psi_wall
    psi = cwp.psi.copy()

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
        drift(cwp, angle="theta", theta_lim=theta_lim, canvas=canvas)
        logger.debug("\tPlotting particle's Pθ drift.")

    # Set psi limits (Normalised to psi_wall)
    if type(psi_lim) is str:
        if psi_lim == "auto":
            psi_diff = psi.max() - psi.min()
            psi_mid = (psi.max() + psi.min()) / 2
            psi_lower = max(0, psi_mid - 0.6 * psi_diff)
            psi_higher = psi_mid + 0.6 * psi_diff
            psi_lim = np.array([psi_lower, psi_higher])
            logger.debug("\tUsing automatic ψ limits.")
    else:
        psi_lim = np.array(psi_lim) * psi_wall
        logger.debug("\tUsing user-defined ψ limits.")
    psi_min = psi_lim[0]
    psi_max = psi_lim[1]

    # Calculate Energy values
    grid_density = config["contour_grid_density"]
    theta, psi = np.meshgrid(
        np.linspace(theta_min, theta_max, grid_density),
        np.linspace(psi_min, psi_max, grid_density),
    )
    values = _calcW_grid(cwp, theta, psi, Pzeta0, contour_Phi, units)
    span = np.array([values.min(), values.max()])
    logger.debug(f"\tEnergy values span from {span[0]:.4g}{units} to {span[1]:.4g}{units}.")

    # Create Figure
    if levels is None:  # If non is given
        levels = config["contour_levels"]
        logger.debug("\tUsing default number of levels.")
    else:
        logger.debug(f"\tOverwritting default levels number to {levels}")
    contour_kw = {
        "vmin": span[0],
        "vmax": span[1],
        "levels": levels,
        "cmap": config["contour_cmap"],
        "zorder": 1,
    }

    # Contour plot
    C = ax.contourf(theta, psi / psi_wall, values, **contour_kw)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\psi/\psi_{wall}$", rotation=90)
    ticks = ["-2π", "-3π/2", "-π", "-π/2", "0", "π/2", "π", "3π/2", "2π"]
    plt.xticks(np.linspace(-2 * np.pi, 2 * np.pi, 9), ticks)
    ax.set(xlim=[theta_min, theta_max], ylim=np.array(psi_lim) / psi_wall)
    ax.set_facecolor("white")

    if wall_shade:  # ψ_wall boundary rectangle
        rect = Rectangle((theta_lim[0], 1), 2 * np.pi, psi_max / psi_wall, alpha=0.2, color="k")
        ax.add_patch(rect)
        logger.debug("\tAdding wall shade.")

    if not kwargs:  # If called for a single particle
        cbar = fig.colorbar(C, ax=ax, fraction=0.03, pad=0.2, label=f"E[{units}]")
        cbar_kw = {
            "linestyle": "-",
            "zorder": 3,
            "color": config["cbar_color"],
        }
        E_cbar = _cbar_energy(cwp, units)
        cbar.ax.plot([0, 1], [E_cbar, E_cbar], **cbar_kw)
        logger.debug(f"\tSingle particle call. Adding energy label at {E_cbar:.4g}{units}")

    if not kwargs:  # If called for a single particle
        fig.set_tight_layout(True)
        plt.ion()
        plt.show(block=True)
        logger.info("--> Energy contour successfully plotted (returned null)\n")
    elif kwargs:  # If called for a collection
        logger.info("--> Energy contour successfully plotted (returned contour object)\n")
        return C


# ---------------------------------------------------------------------------


def _calcW_grid(
    cwp,
    theta: np.array,
    psi: np.array,
    Pzeta: float,
    contour_Phi: bool,
    units: str,
):
    r"""Returns a single value or a grid of the calculated Hamiltonian.

    Only to be called internally, by ``contour_energy()``..

    Args:
        theta (np.array): The :math:`\theta` values.
        psi (np.array): The :math:`\psi` values.
        Pzeta (float): The :math:`P_\zeta` value.
        contour_Phi (bool): Whether or not to add the electric potential term
            :math:`e\Phi`.
        units (str): The energy units.
    """
    if contour_Phi:
        logger.debug("\tAdding Φ term to the contour.")
    logger.debug(f"\tCalculating energy values in a {theta.shape} grid.")

    # Get all needed attributes first
    mass_amu = cwp.mass_amu
    sign = cwp.sign
    mu = cwp.mu
    NU_to_eV = cwp.NU_to_eV
    Volts_to_NU = cwp.Volts_to_NU
    q = cwp.q
    Bfield = cwp.Bfield
    Efield = cwp.Efield

    r = np.sqrt(2 * psi)
    B = Bfield.B(r, theta)
    psip = q.psip_of_psi(psi)

    W = (Pzeta + psip) ** 2 * B**2 / (2 * Bfield.g**2 * mass_amu) + mu * B  # Without Φ

    # Add Φ if asked
    if contour_Phi:
        Phi = Efield.Phi_of_psi(psi)
        Phi *= Volts_to_NU * sign
        W += Phi  # all normalized

    if units == "eV":
        W *= NU_to_eV
        logger.debug("\tPlotting energy levels in [eV]")
    elif units == "keV":
        W *= NU_to_eV / 1000
        logger.debug("\tPlotting energy levels in [keV]")
    else:
        logger.debug("\tPlotting energy levels in [NU]")

    return W


# ---------------------------------------------------------------------------


def _cbar_energy(cwp, units):
    """Returns the height of the energy colorbar label.

    Args:
        units (str): The energy units.

    Returns:
        float: The energy value.
    """
    logger.debug("Calling _cbar_energy()")

    if units == "normal":
        E_cbar = cwp.E
    elif units == "eV":
        E_cbar = cwp.E_eV
    elif units == "keV":
        E_cbar = cwp.E_eV / 1000
    else:
        print('units must be either "normal", "eV" or "keV"')
        return 0

    return E_cbar
