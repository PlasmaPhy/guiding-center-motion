"""Guiding Center Motion

.. rubric:: Classes
.. autosummary::
    :toctree:
    
    Particle
    Collection
    Plot
    Plots

.. rubric:: Modules
.. autosummary::
    :toctree:

    bfield
    efield
    qfactor

"""

from .particle import Particle
from .collection import Collection
from .plot import Plot
from .plots import Plots
from .animate import animate

from . import bfield, qfactor, efield, parabolas

__all__ = [
    "Particle",
    "Collection",
    "Plot",
    "Plots",
    "bfield",
    "efield",
    "qfactor",
    "parabolas",
    "animate",
]
