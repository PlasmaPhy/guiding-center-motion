"""
This module creates many particles, reading initial conditions 
and configurations from the ``../parameters.py`` file.
"""

import numpy as np
from .particle import Particle
from .plots import Plots


class Collection:

    def __init__(self, file):
        """Initialized Collection class.

        This class is a *Collection* of many particles. It reads their initial
        conditions form the ``./parameters.py`` file, calculates their attributes
        and orbits, and calls upon the class ``Plots`` to plot the results.

        Args:
            file (.py file): The python file containing the parameters.
        """
        # Check data
        self.params = file.params
        self.freq_analyzed = False
        self._check(file)
        self._create()

    def _check(self, file) -> bool:
        """Checks for the validity of the parameters file.

        Checks if all given parameter lists are of length 1 or otherwise of
        equal length with each other, and if their values are valid.

        Also creates truth flags for every parameter: True if single-valued,
        and False if variable.

        Args:
            file (.py file): The python file containing the parameters.

        Returns:
            bool: Truth value depending on the validity of the file
        """

        print("Checking Data...")

        # Store the lenghts of each parameter and find the number of particles.
        self.lengths = {x: 1 for x in self.params}
        for key, value in self.params.items():
            if isinstance(value, (int, float)) or key == "t_eval":
                continue
            if isinstance(value, (list, np.ndarray)):
                self.lengths[key] = len(value)

        self.n = max(self.lengths.values())

        # Check for multiple efields, bfields, qs, species and create flags
        for key, value in self.lengths.items():
            exec("self.multiple_" + key + "=bool(" + str(value) + "-1)")

        # Check lengths and print results
        if all((_ == self.n) or (_ == 1) for _ in self.lengths.values()):
            print(f"Data is OK. Number of particles = {self.n}")
        else:
            print("Error: Multiple valued parameters must all be of same length")
            return False

    def _create(self):
        """Initiates the particles."""

        # Make an iterable copy of params
        # CAUTION! all objects of same value point to one single object.
        # Changing one of them changes all of them.
        params = self.params.copy()
        for key, value in params.items():
            if self.lengths[key] == 1:
                params[key] = [value] * self.n

        self.particles = []

        for i in range(self.n):
            R, a = params["R"][i], params["a"][i]  # Major/Minor Radius in [m]
            q = params["q"][i]
            Bfield = params["Bfield"][i]
            Efield = params["Efield"][i]

            # Create Particle
            species = params["species"][i]
            mu = params["mu"][i]  # Magnetic moment
            theta0 = params["theta0"][i]
            psi0 = params["psi0"][i]  # times psi_wall
            z0 = params["z0"][i]
            Pz0 = params["Pz0"][i]
            t_eval = params["t_eval"][i]  # t0, tf, steps

            init_cond = [theta0, psi0, z0, Pz0]

            # Particle Creation
            p = Particle(species, mu, init_cond, t_eval, R, a, q, Bfield, Efield)
            self.particles.append(p)

    def run_all(self, orbit=True):
        """Calculates all the particle's orbit, by running Particle.run() itself.

        Some plots and calculations, such as the parabolas and the orbit type
        calculation don't require the whole orbit to be calculated, since they
        only depend on the initial conditions. We can therefore save valuable time.

        Args:
            orbit (bool, optional): Whether or not to calculate the particles'
                orbits. Defaults to True.
        """

        for p in self.particles:
            p.run(info=False, orbit=orbit)

        # Create plot instance
        self.plot = Plots(self)

    # Added in order to use ωθ from events and FFT in def omega_thetas in plots.py
    def freq_analysis_all(self, angle="theta"):

        for p in self.particles:
            print(f"Analyzed particle with Pz0: {p.Pz0}")
            p.freq_analysis(angle=angle, info=False, plot=False)

        self.freq_analyzed = True
        print(self.freq_analyzed)
