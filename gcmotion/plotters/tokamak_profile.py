import numpy as np
import matplotlib.pyplot as plt

from gcmotion.utils._logger_setup import logger

from gcmotion.configuration.plot_parameters import tokamak_profile as config


def tokamak_profile(cwp, zoom: list = [0, 1.1]):
    r"""Plots the electric field, potential, and q factor,
    with respect to :math:`\psi/\psi_{wall}`.

    Args:
        zoom (list, optional): zoom to specific area in the x-axis of the electric field
            and potential plots. Defaults to [0, 1.1].
    """
    logger.info("Plotting tokamak profile...")

    # Get all needed attributes first
    a = cwp.a
    psi_wall = cwp.psi_wall
    q = cwp.q
    Bfield = cwp.Bfield
    Efield = cwp.Efield

    fig = plt.figure(figsize=(12, 8))
    fig.subplots_adjust(hspace=0.5)
    ax_phi = fig.add_subplot(321)
    ax_E = fig.add_subplot(322)
    ax_q1 = fig.add_subplot(323)
    ax_q2 = fig.add_subplot(324)
    # fig.tight_layout()

    psi = np.linspace(0, 1.1 * psi_wall, 1000)
    Er = Efield.Er_of_psi(psi)
    Phi = Efield.Phi_of_psi(psi)

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

        # Radial E field
        ax_phi.plot(psi / psi_wall, Er, color="b", linewidth=1.5)
        ax_phi.plot([1, 1], [Er.min(), Er.max()], color="r", linewidth=1.5)
        ax_phi.set_xlabel(r"$\psi/\psi_{wall}$")
        ax_phi.set_ylabel(E_ylabel)
        ax_phi.set_title("Radial electric field [kV/m]", c="b")

        # Electric Potential
        ax_E.plot(psi / psi_wall, Phi, color="b", linewidth=1.5)
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
        y1 = q.q_of_psi(psi)
        if type(y1) is int:  # if q = Unity
            y1 *= np.ones(psi.shape)
        ax_q1.plot(psi / psi_wall, y1, color="b", linewidth=1.5)
        ax_q1.plot([1, 1], [y1.min(), y1.max()], color="r", linewidth=1.5)
        ax_q1.set_xlabel(r"$\psi/\psi_{wall}$")
        ax_q1.set_ylabel(r"$q(\psi)$")
        ax_q1.set_title(r"$\text{q factor }q(\psi)$", c="b")

        # ψ_π(ψ)
        y2 = q.psip_of_psi(psi)
        ax_q2.plot(psi / psi_wall, y2, color="b", linewidth=1.5)
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
        ax_B.set_yticks([0, a])

        rs = np.linspace(0, a, 100)
        thetas = np.linspace(0, 2 * np.pi, 100)
        r, theta = np.meshgrid(rs, thetas)
        B = Bfield.B(r, theta)
        levels = config["contour_params"]["levels"]
        cmap = config["contour_params"]["cmap"]
        ax_B.contourf(theta, r, B, levels=levels, cmap=cmap)

        logger.debug("\t-> Magnetic field profile successfully plotted.")

    plot_electric()
    plot_q()
    plot_magnetic()

    fig.set_tight_layout(True)
    plt.ion()
    plt.show(block=True)

    logger.info("--> Tokamak profile successfully plotted.\n")
