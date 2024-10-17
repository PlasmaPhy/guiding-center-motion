"""
Creates the q-factor profile of the system
==========================================

Creates the q factor profile of the system.

To add a new q-factor, simply copy-paste an already existing class
(idealy the Unity one) and fill the ``q_of_psi()`` and ``psip_of_psi()`` 
methods to fit your q-factor. In case your q factor has extra parameters
you want to pass as arguments, you must also create an ``__init__()``
method and declare them. To avoid errors, your class should inherit the
``QFactor`` class.

.. note::
    Keep in mind that when those methods return singular values (rather than
    np.ndarrays), they should return a float, and not a np.float. This is mainly
    for optimization reasons and should probably not cause problems.

.. note::
    You can create new q factor profiles in other .py files as well, but
    you have to specify the base class path and import them correctly 
    as well.

The general structure is this::

    class MyQFactor(QFactor):

        def __init__(self, *<parameters>):
            self.id = "foo" # Simple id used only for logging.
            self.params = {} # Tweakable parameters, used only for logging.
            <set parameters>

        def q_of_psi(self, psi):
            return q

        def psip_of_psi(self, psi):
            return pisp

"""

import numpy as np
from math import atan
from scipy.special import hyp2f1
from abc import ABC, abstractmethod


class QFactor(ABC):
    """q Factor base class

    .. note::
        This class does nothing, it is only a template.
    """

    @abstractmethod
    def __init__(self):
        self.id = "Base Class"
        self.params = {}

    @abstractmethod
    def q_of_psi(self, psi: float | list | np.ndarray) -> float | list | np.ndarray:
        """Calculates q(ψ). Return type should be same as input.

        Used inside dSdt, Φ derivatives (returns a float) and plotting of
        q factor (returns an np.ndarray).

        :param psi: Value(s) of ψ.
        :returns: Calculated q(ψ)
        """
        pass

    @abstractmethod
    def psip_of_psi(self, psi: float | np.ndarray) -> float | np.ndarray:
        r"""Calculates :math:`\psi_p(\psi)`.

        Used in calculating :math:`\psi_{p,wall}` in many methods (returns a float), in calculating
        :math:`\psi_p`'s time evolution (returns an np.ndarray), in Energy contour calculation
        (returns an np.ndarray) and in q-factor plotting (returns an np.ndarray)

        :param psi: Value(s) of ψ.
        :returns: Calculated :math:`\psi_p(\psi)`.
        """
        pass


# ====================================================


class Unity(QFactor):
    r"""Initializes an object q with :math:`q(\psi) = 1`"""

    def __init__(self):
        self.id = "Unity"
        self.params = {}
        return

    def q_of_psi(self, psi):
        return 1

    def psip_of_psi(self, psi):
        return psi


class Parabolic(QFactor):
    r"""Initializes an object q with :math:`q(\psi) = 1 + \psi^2`"""

    def __init__(self):
        self.id = "Parabolic"
        self.params = {}
        return

    def q_of_psi(self, psi):
        return 1 + psi**2

    def psip_of_psi(self, psi):
        return atan(psi)


class Hypergeometric(QFactor):
    r"""Initializes an object q with
    :math:`q(\psi) = q_0\\bigg[ 1 + \\bigg( \dfrac{\psi}{\psi_k(q_{wall})} \\bigg)^n \\bigg]^{1/n}`.
    """

    def __init__(self, R: float, a: float, q0: float = 1.1, psi_knee: float = 2.5, n: int = 2):
        """Parameters initialization.

        :param R: The tokamak's major radius.
        :param a: The tokamak's minor radius.
        :param q0: q-value at the magnetic axis. (Optional, defaults to 1.1)
        :param psi_knee: Location of knee. (Optional, defaults to 2.5)
        :param n:  Order of equillibrium (1: peaked, 2: round, 4: flat).
            (Optional, defaults to 2)
        """
        self.id = "Hypergeometric"
        self.params = {"q0": q0, "psi_knee": psi_knee, "n": n}

        self.r_wall = a / R  # normalized to R
        self.psi_wall = (self.r_wall) ** 2 / 2
        self.psi_wall = self.psi_wall
        self.q0 = q0
        self.psi_knee = 0.75 * self.psi_wall
        self.n = n
        self.q_wall = self.q_of_psi(self.psi_wall)

    def q_of_psi(self, psi):
        return self.q0 * (1 + (psi / (self.psi_knee)) ** self.n) ** (1 / self.n)

    def psip_of_psi(self, psi):
        a = b = 1 / self.n
        c = 1 + 1 / self.n
        z = (1 - (self.q_wall / self.q0) ** self.n) * (psi / self.psi_wall) ** self.n
        if type(psi) is float:
            return psi / self.q0 * float(hyp2f1(a, b, c, z))
        else:
            return psi / self.q0 * hyp2f1(a, b, c, z)
