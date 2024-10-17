import numpy as np
from scipy.integrate import solve_ivp
from math import sqrt

from gcmotion.utils._logger_setup import logger


def orbit(
    t: np.ndarray,
    init_cond: list,
    constants: dict,
    profile: dict,
    events: list,
):

    logger.info(f"Calculating orbit with events {events}")

    # Tokamak profile
    q = profile["q"]
    Bfield = profile["Bfield"]
    Efield = profile["Efield"]
    Volts_to_NU = float(profile["Volts_to_NU"])

    # Constants of motion
    # E = float(constants["E"]) # Not used
    mu = float(constants["mu"])
    # Pzeta0 = float(constants["Pzeta0"]) # Not used

    # Initial Conditions
    S0 = [
        float(init_cond["theta0"]),
        float(init_cond["psi0"]),
        float(init_cond["zeta0"]),
        float(init_cond["rho0"]),
    ]

    def dSdt(t, S):
        """Sets the diff equations system to pass to scipy.

        All values are in normalized units (NU).
        """

        theta, psi, z, rho = S

        # Intermediate values
        phi_der_psip, phi_der_theta = Efield.Phi_der(psi)
        phi_der_psip *= Volts_to_NU
        phi_der_theta *= Volts_to_NU
        B_der_psi, B_der_theta = Bfield.B_der(psi, theta)
        q_value = q.q_of_psi(psi)
        r = sqrt(2 * psi)
        B = Bfield.B(r, theta)
        par = mu + rho**2 * B
        bracket1 = -par * q_value * B_der_psi + phi_der_psip
        bracket2 = par * B_der_theta + phi_der_theta
        D = Bfield.g * q_value + Bfield.I

        # Canonical Equations
        theta_dot = 1 / D * rho * B**2 + Bfield.g / D * bracket1
        psi_dot = -Bfield.g / D * bracket2 * q_value
        rho_dot = psi_dot / (Bfield.g * q_value)
        z_dot = rho * B**2 / D - Bfield.I / D * bracket1

        return [theta_dot, psi_dot, z_dot, rho_dot]

    t_span = (t[0], t[-1])
    sol = solve_ivp(
        dSdt,
        t_span=t_span,
        y0=S0,
        t_eval=t,
        rtol=1e-8,
        events=events,
        dense_output=True,
    )

    theta = sol.y[0]
    psi = sol.y[1]
    zeta = sol.y[2]
    rho = sol.y[3]
    t_eval = sol.t
    t_events = sol.t_events
    y_events = sol.y_events
    message = f"{sol.status}: {sol.message}"

    logger.debug(f"Solver status: {message}")

    # Calculate psip and Canonical Momenta
    psip = q.psip_of_psi(psi)
    Ptheta = psi + rho * Bfield.I
    Pzeta = rho * Bfield.g - psip

    solution = {
        "theta": theta,
        "psi": psi,
        "zeta": zeta,
        "rho": rho,
        "psip": psip,
        "Ptheta": Ptheta,
        "Pzeta": Pzeta,
        "t_eval": t_eval,
        "t_events": t_events,
        "y_events": y_events,
        "message": message,
    }

    return solution
