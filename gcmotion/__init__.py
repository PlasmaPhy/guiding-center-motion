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

from gcmotion.plotters import plot

__all__ = [
    "_logger_setup",
    "Particle",
    "collection",
    "qfactor",
    "bfield",
    "efield",
    "plot",
]
