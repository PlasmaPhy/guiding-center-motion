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

# Import the logger first
from gcmotion.utils import _logger_setup


from gcmotion.tokamak_config import qfactor, bfield, efield

from gcmotion.classes.particle import Particle

from gcmotion.plotters.time_evolution import time_evolution
from gcmotion.plotters.tokamak_profile import tokamak_profile
from gcmotion.plotters.drift import drift
from gcmotion.plotters.drifts import drifts
from gcmotion.plotters.contour_energy import contour_energy

__all__ = [
    "_logger_setup",
    "Particle",
    "collection",
    "qfactor",
    "bfield",
    "efield",
    "time_evolution",
    "tokamak_profile",
    "drift",
    "drifts",
    "contour_energy",
]
