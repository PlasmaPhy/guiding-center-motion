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

__all__ = [
    "_logger_setup",
    "Particle",
    "collection",
    "qfactor",
    "bfield",
    "efield",
    "time_evolution",
    "tokamak_profile",
]
