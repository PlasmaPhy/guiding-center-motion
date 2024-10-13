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
    freq

"""

from loguru import logger

from .particle import Particle
from .collection import Collection
from .plot import Plot
from .plots import Plots
from .animate import animate

from . import bfield, qfactor, efield, parabolas, freq

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
    "freq",
]

# Setup logger
logger.remove()

# Format templates
fmt = "{time:HH:mm:ss:SSS} | {function: <20} |  {level: ^7} | {message}"
# fmt = "{time:HH:mm:ss:SSS} | {name: <18} |  {level: >6} | {message}"
level = "DEBUG"

logger.add("log.txt", delay=True, level=level, format=fmt, mode="w")
logger.info(f"Logger added on {level} level.\n")
logger.info("---------------------------------------------------")
