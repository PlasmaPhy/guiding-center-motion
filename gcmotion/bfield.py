"""
Creates the Magnetic field of the system
========================================

Creates the Magnetic field of the system.

To add a new Magnetic Field, simply copy-paste an already existing class
(idealy the Nofield one) and fill the ``__init__()`` and ``B()`` methods 
to fit your Magnetic Field.  In case your Magnetic field has extra 
parameters you want to pass as arguments, you must also create an 
``__init__()`` method and declare them. To avoid errors, your class should
inherit the ``MagneticField`` class.

.. danger::
    **All values, both input and output are in SI units.**

    Specifically, g and I in [NU] and B in [T].

.. note::
    The B method returns either a float or an np.ndarray, depending on the
    input type. Solvers need to be fast so they work with built-in floats,
    while plotting functions work with np.ndarrays.

.. note::
    You can create new Magnetic fields in other .py files as well, but
    you have to specify the base class path and import them correctly 
    as well.

The general structure is this::

    class MyMagneticField(MagneticField):

        def __init__(self, *<parameters>):
            self.id = "foo" # Simple id used only for logging.
            self.params = {} # Tweakable parameters, used only for logging.
            <set parameters>
    
        def B(self, r, theta): 
            if type(r) is float:
                return 1 - r * cos(theta)
            else:
                return 1 - r * np.cos(theta)
"""

import numpy as np
from math import cos
from abc import ABC, abstractmethod


class MagneticField(ABC):
    r"""Electric field base class

    .. note::
        This class does nothing, it is only a template.
    """

    def __init__(self):
        self.id = "Base Class"
        self.params = {}

    @abstractmethod
    def B(self, r: float | np.ndarray, theta: float | np.ndarray):
        r"""Returns the magnetic field strength.

        Used inside the solver (input and return type must be float),
        and the countour plot of the magnetic field profile (input and
        return type must be np.array).

        Args:
            r (float, np.ndarray): the r position of the particle.
            theta (float, np.ndarray): the theta position of the particle

        Returns:
            B (float. np.ndarray): The magnetic field strength.
        """
        pass


# =======================================================


class LAR(MagneticField):
    """Initializes the standard Large Aspect Ration magnetic field."""

    def __init__(self, i: float, g: float, B0: float):
        r"""Parameters initialization.

        Args:
            i (float): The toroidal current.
            g (float): The poloidal current.
            B0 (float): The magnetic field strength on the
                magnetic axis.
        """
        self.id = "LAR"
        self.params = {"I": i, "g": g, "B0": B0}

        self.I, self.g, self.B0 = i, g, B0
        self.is_lar = True

    def B(self, r: float | np.ndarray, theta: float | np.ndarray):
        if isinstance(r, (int, float)):
            return 1 - r * cos(theta)
        else:
            return 1 - r * np.cos(theta)
