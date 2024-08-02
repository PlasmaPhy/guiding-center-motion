""" To add a new Electric Field, simply copy-paste an already existing class
    (idealy the Nofield one) and fill the __init__() method with the 
    parameters, and the Phi_der() and Er_of_psi(), and Phi_of_psi()
    to fit your Electric Field.
 
    Keep in mind that when those methods return singular values (rather than
    np.ndarrays), they should return a float, and not a np.float. This is mainly
    for optimization reason and should probably not cause problems.

    ALL VALUES, BOTH INPUT PARAMETERS AND OUTPUT ARE IN SI UNITS, specifically 
    [V/m] and [V].

    The general structure is this:

    class <name>:

        def __init__(self, **<parameters>):
            <set parameters>

        def Phi_der(self, psi): 
            > Derivatives of Φ(ψ) with respect to ψ_π, θ, in [V].
            > 
            > Intended for use only inside the ODE solver. Returns the potential
            > in [V], so the normalisation is done inside the solver.
            > Args:
            >     psi(float): the magnetic flux surface.
            > Returns:
            >     list: list containing the calculated derivatives as floats
            
            return [Phi_der_psip, Phi_der_theta]

        def Er_of_psi(self, psi):
            > Calculates radial Electric field component in [V/m] from ψ.
            > 
            > Used for plotting the Electric field
            > 
            > Args:
            >     psi (np.ndarray): The ψ values.
            > 
            > Returns:
            >     np.ndarray: Numpy array with calculated E values.
            > 
            r = np.sqrt(2 * psi)
            return E

        def Phi_of_psi(self, psi):
        > Calculates Electric Potential in [V] from ψ.
        > 
        > Used for plotting the Electric Potential, the particles initial Φ,
        > and the Φ values for the contour plot.
        > 
        > Args:
        >     psi (np.ndarray): The ψ values.
        > 
        > Returns:
        >     np.ndarray: Numpy array with calculated values. 

        return Phi
    """

import numpy as np
from scipy.special import erf
from math import sqrt, exp


class Nofield:
    """Initializes an electric field of 0

    Exists to avoid compatibility issues.
    """

    def Phi_der(self, psi):
        return [0, 0]

    def Er_of_psi(self, psi):
        return 0 * psi

    def Phi_of_psi(self, psi):
        return 0 * psi


class Parabolic:
    """Initializes an electric field of the form E = ar^2 + b"""

    def __init__(self, R, a, q, alpha=1, beta=0):
        self.a = alpha
        self.b = beta
        self.q = q
        self.r_wall = a / R
        self.psi_wall = (self.r_wall) ** 2 / 2  # normalized to R
        self.psip_wall = q.psip_of_psi(self.psi_wall)

    def Phi_der(self, psi):
        r = np.sqrt(2 * psi)
        Phi_der_psip = -self.q.q_of_psi(psi) * (self.a * r - self.b / r)
        Phi_der_theta = 0
        return [Phi_der_psip, Phi_der_theta]

    def Er_of_psi(self, psi):
        r = np.sqrt(2 * psi)
        E = self.a * r**2 + self.b
        return E

    def Phi_of_psi(self, psi):
        r = np.sqrt(2 * psi)
        Phi = -(self.a / 3) * r**3 - self.b * r
        return Phi


class Radial:
    """Initializes an electric field of the form:
    E(r) = -Ea_norm*exp(-(r-r_a)^2 / r_w^2))
    """

    def __init__(self, R, a, q, Ea=75000):

        self.q = q
        self.r_wall = a / R
        self.psi_wall = (self.r_wall) ** 2 / 2  # normalized to R
        self.psip_wall = q.psip_of_psi(self.psi_wall)

        self.Ea = Ea  # V/m
        self.r0 = sqrt(2 * self.psi_wall)
        self.ra = 0.98 * self.r0  # Defines the minimum point
        self.Efield_min = self.ra**2 / 2
        self.rw = self.r0 / 50  # waist, not wall
        self.psia = self.ra**2 / 2
        self.psiw = self.rw**2 / 2  # waist, not wall

        # Square roots, makes it a bit faster
        self.sr_psia = sqrt(self.psia)
        self.sr_psiw = sqrt(self.psiw)

    def Phi_der(self, psi):
        Phi_der_psip = (
            self.q.q_of_psi(psi)
            * self.Ea
            / (sqrt(2 * psi))
            * exp(-((sqrt(psi) - self.sr_psia) ** 2) / self.psiw)
        )
        Phi_der_theta = 0

        return [Phi_der_psip, Phi_der_theta]

    def Er_of_psi(self, psi):
        r = np.sqrt(2 * psi)
        E = -self.Ea * np.exp(-((r - self.ra) ** 2) / self.rw**2)
        return E

    def Phi_of_psi(self, psi):
        Phi = (
            self.Ea
            * np.sqrt(np.pi * self.psiw / 2)
            * (erf((np.sqrt(psi) - self.sr_psia) / self.sr_psiw) + erf(self.sr_psia / self.sr_psiw))
        )
        return Phi
