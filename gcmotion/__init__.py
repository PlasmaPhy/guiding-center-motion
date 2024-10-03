"""Guiding Center Motion

.. rubric:: Classes
.. autosummary::
    :toctree:

    Particle
    Particles
    Plot

.. rubric:: Modules
.. autosummary::
    :toctree:

    efield
    qfactor
    parabolas
"""

from .particle import Particle
from .particles import Particles
from .plot import Plot

from . import qfactor, efield, parabolas

from .animate import animate

__all__ = ["Particle", "Particles", "Plot", "efield", "qfactor", "parabolas", "animate"]
