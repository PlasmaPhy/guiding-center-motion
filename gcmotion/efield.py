"""
Creates the Electric field of the system
========================================

Creates the Electric field of the system.

To add a new Electric Field, simply copy-paste an already existing class
(idealy the Nofield one) and fill the ``Phi_der()`` and ``Er_of_psi()``, 
and ``Phi_of_psi()`` methods to fit your Electric Field.  In case your 
Electric field has extra parameters you want to pass as arguments, you 
must also create an ``__init__()`` method and declare them. To avoid 
errors, your class should inherit the ``ElectricField`` class.

.. danger::
    **All values, both input and output are in SI units.**

    Specifically [V/m] and [V].

.. note::
    Keep in mind that when those methods return singular values (rather than
    np.ndarrays), they should return a float, and not a np.float. This is mainly
    for optimization reasons and should probably not cause problems.

.. note::
    You can create new Electric fields in other .py files as well, but
    you have to specify the base class path and import them correctly 
    as well.

The general structure is this::

    class MyElectricField(ElectricField):

        def __init__(self, *<parameters>):
            <set parameters>
    
        def Phi_der(self, psi): 
            return [Phi_der_psip, Phi_der_theta]
    
        def Er_of_psi(self, psi):
            r = np.sqrt(2 * psi)
            return E
    
        def Phi_of_psi(self, psi):
            return Phi
"""

import numpy as np
from scipy.special import erf
from .qfactor import QFactor
from math import sqrt, exp
from abc import ABC, abstractmethod


class ElectricField(ABC):
    """Electric field base class

    .. note::
        This class does nothing, it is only a template.
    """

    def __init__():
        """Not used, each class must define its own."""

    @abstractmethod
    def Phi_der(self, psi: float) -> tuple[float, float]:
        r"""Derivatives of Φ(ψ) with respect to :math:`\psi_p, \\theta` in [V].

        Intended for use only inside the ODE solver. Returns the potential
        in [V], so the normalisation is done inside the solver.

        :param psi: The magnetic flux surface.
        :return: List containing the calculated derivatives as floats
        """
        pass

    @abstractmethod
    def Er_of_psi(self, psi: np.ndarray) -> np.ndarray:
        """Calculates radial Electric field component in [V/m] from ψ.

        Used for plotting the Electric field

        :param psi: The ψ values.
        :returns: Numpy array with calculated E values.
        """
        pass

    @abstractmethod
    def Phi_of_psi(self, psi: np.ndarray) -> np.ndarray:
        """Calculates Electric Potential in [V] from ψ.

        Used for plotting the Electric Potential, the particles initial Φ,
        and the Φ values for the contour plot.

        :param psi: The ψ values.
        :returns: Numpy array with calculated values.
        """
        pass


# =======================================================


class Nofield(ElectricField):
    """Initializes an electric field of 0

    Exists to avoid compatibility issues.
    """

    def __init__(self):
        """Not used."""
        return

    def Phi_der(self, psi):
        return (0, 0)

    def Er_of_psi(self, psi):
        return 0 * psi

    def Phi_of_psi(self, psi):
        return 0 * psi


class Parabolic(ElectricField):
    """Initializes an electric field of the form: :math:`E(r) = ar^2 + b` (BETA)"""

    def __init__(self, R: float, a: float, q: QFactor, alpha: float = 1, beta: float = 0):
        """Parameters initialization.

        :param R: The tokamak's major radius.
        :param a: The tokamak's minor radius.
        :param q: q factor profile.
        :param alpha: The :math:`r^2` coefficient. (Optional, defaults to 1)
        :param beta: The constant coefficient. (Optional, defaults to 0)
        """
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


class Radial(ElectricField):
    r"""Initializes an electric field of the form:
    :math:`E(r) = -E_a\exp\bigg[-\dfrac{(r-r_a)^2}{r_w^2})\bigg]`
    """

    def __init__(
        self,
        R: float,
        a: float,
        q: QFactor,
        Ea: float = 75000,
        minimum: float = 0.7,
        waist_width: float = 10,
    ):
        r"""Parameters initialization.

        :param R: The tokamak's major radius.
        :param a: The tokamak's minor radius.
        :param q: q factor profile.
        :param Ea: The Electric field magnitude. (Optional, defaults to 75000)
        :param beta: The constant coefficient. (Optional, defaults to 0)
        :param minimum: The Electric field's minimum point with respect to
            :math:`\psi_{wall}`.
        :param waist_width: The Electric field's waist width, defined as:
            :math:`r_w = \dfrac{a}{\\text{waste width}}`
        """

        self.q = q
        self.r_wall = a / R
        self.psi_wall = (self.r_wall) ** 2 / 2  # normalized to R
        self.psip_wall = q.psip_of_psi(self.psi_wall)
        self.minimum = minimum
        self.waist_width = waist_width

        self.Ea = Ea  # V/m
        self.ra = self.minimum * self.r_wall  # Defines the minimum point
        self.Efield_min = self.ra**2 / 2
        self.rw = self.r_wall / self.waist_width  # waist, not wall
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
