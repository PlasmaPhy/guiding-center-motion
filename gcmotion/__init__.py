"""Guiding Center Motion

.. rubric:: Classes
.. autosummary::
    :toctree:

    Particle
    Particles

.. rubric:: Modules
.. autosummary::
    :toctree:

    efield
    qfactor
    parabolas
"""

from .particle import Particle
from .particles import Particles

from . import qfactor, efield, parabolas

from .animate import animate

__all__ = ["Particle", "Particles", "efield", "qfactor", "parabolas", "animate"]
