"""
This module initializes the "Particle" class, which calculates the orbit,
orbit type, and can draw several different plots
"""

import numpy as np
from time import time
from math import sqrt
from .plot import Plot
from .parabolas import Construct
from .bfield import MagneticField
from .efield import ElectricField, Nofield
from .qfactor import QFactor
from . import config, logger

from .scripts.orbit import orbit


class Particle:
    """Initializes a particle, which calculates the orbit,
    orbit type, unit convertion factors, and motion frequencies.

    Supposedly there is no need to change anything here. Electric Fields and Q factors
    should be changed in the respective ``gcmotion/*.py`` files.
    """

    def __init__(
        self,
        species: str,
        mu: float,
        init_cond: np.array,
        t_eval: np.array,
        R: float,
        a: float,
        q: QFactor,
        Bfield: MagneticField,
        Efield: ElectricField,
        rtol: float = 10e-8,
    ):
        r"""Initializes particle and grabs configuration.

        Args:
            species (str): the particle species, used to later set charge and mass
                automatically (from ``gcmotion/config.yaml``)
            mu (float): magnetic moment
            init_cond (np.array): 1x4 initial conditions array (later, self.init_cond
                includes 2 more initial conditions,
                [:math:`\theta_0, \psi_0, \psi_{p0}, \zeta_0, P_{\zeta_0}`])
            t_eval (np.array): The ODE interval, in [:math:`t_0, t_f`, steps]
            R (float): The tokamak's major radius in [m].
            a (float): The tokamak's minor radius in [m].
            q (QFactor): Qfactor object that supports query methods for getting values
                of :math:`q(\psi)` and :math:`\psi_p(\psi)`.
            Bfield (MagneticField): Magnetic Field Object that supports query methods for getting
                values for the field itself.
                magnitude (in [T]) of the magnetic field B, as in
                :math:`[g, I, B_0]`.
            Efield (ElectricField): Electric Field Object that supports query methods for getting
                values for the field itself and some derivatives of
                its potential.
            rtol (float): Relative tolerance of the RK45 solver. Defaults to :math:`10^{-6}`.
        """
        logger.info("--------Initializing particle--------")

        def grab_configuration():
            """Attempt to import the config Dictionary."""

            logger.info("Attempting to grab configuration.")
            try:
                self.configs = config.configs
                logger.info("--> Grabbed configuration successfully.")
            except (IOError, ValueError, NameError, OSError):
                logger.error("--> Failed grabbing configuration.")

        def setup_tokamak():
            """Sets up tokamak-related attributes."""

            logger.info("Setting up Tokamak...")

            # Dimensions
            self.R, self.a = R, a

            logger.debug(f"\tTokamak dimensions: R = {self.R}, a = {self.a}")

            # Objects
            self.q = q
            self.Bfield = Bfield
            if Efield is None or isinstance(Efield, Nofield):
                self.Efield = Nofield()
                self.has_efield = False
            else:
                self.Efield = Efield
                self.has_efield = True

            logger.debug(f"\t'{self.q.id}' qfactor used with parameters {self.q.params}")
            logger.debug(f"\t'{self.Bfield.id}' Bfield used with parameters {self.Bfield.params}")
            logger.debug(f"\t'{self.Efield.id}' Efield used with parameters {self.Efield.params}")

            self.r_wall = self.a / self.R
            self.psi_wall = (self.r_wall) ** 2 / 2  # normalized to R
            self.psip_wall = self.q.psip_of_psi(self.psi_wall)

            # psi_p > 0.5 warning
            if self.psip_wall >= 0.5:
                logger.warning(
                    f"\tWARNING: psip_wall = {self.psip_wall:.5g} >= 0,5."
                    + "Parabolas and other stuff will probably not work"
                )

            logger.debug(
                f"\tDerivative quantities: r_wall = {self.r_wall:.5g}, psi_wall = {self.psi_wall:.5g}"
                + f" (CAUTION: normalised to R), psip_wall = {self.psip_wall:.5g}"
            )

            logger.info("--> Tokamak setup successful.")

        def setup_constants():
            """Grabs particle's constants from ``config.py``"""

            logger.info("Setting up particle's constants...")

            self.species = species
            self.mass_amu = self.configs[self.species + "_mass_amu"]
            self.mass_keV = self.configs[self.species + "_mass_keV"]
            self.mass_kg = self.configs[self.species + "_mass_kg"]
            self.zeta = self.configs[self.species + "_Z"]
            self.e = self.configs["elementary_charge"]
            self.sign = self.zeta / abs(self.zeta)

            logger.debug(f"\tParticle is of species '{self.species}'.")
            logger.info("--> Particle's constants setup successful")

        def setup_solver():
            """Sets up the solver and its parameters"""

            logger.info("Setting up solver parameters...")

            if isinstance(rtol, (int, float)):
                self.rtol = rtol
            else:
                logger.warning("Invalid passed relative tolerance. Using defaults...")
                self.rtol = float(self.configs["rtol"])

            logger.debug(
                f"\tUsing solver method 'RK4(5)', with relative tolerance of {self.rtol} (only used by RK45)."
            )
            logger.info("--> Solver setup successful.")

        def setup_init_cond():
            """Sets up the particles initial condition and parameters, as well as the solver's S0."""

            logger.info("Setting up particle's initial conditions...")

            self.t_eval = t_eval
            self.mu = mu
            self.theta0 = init_cond[0]
            init_cond[1] *= self.psi_wall  # CAUTION! Normalize it to psi_wall
            self.psi0 = init_cond[1]
            self.zeta0 = init_cond[2]
            self.Pzeta0 = init_cond[3]
            self.psip0 = self.q.psip_of_psi(self.psi0)
            self.rho0 = self.Pzeta0 + self.psip0  # Pz0 + psip0
            self.Ptheta0 = self.psi0 + self.rho0 * self.Bfield.I  # psi + rho*I

            logger.debug(
                "ODE initial conditions:\n"
                + f"\ttheta0 = {self.theta0:.5g}, psi0 = {self.psi0:.5g}, zeta0 = {self.zeta0:.5g}, Pzeta0 = {self.Pzeta0:.5g}."
            )
            logger.debug(
                "\tOther initial conditions:\n"
                + f"\tPtheta0 = {self.Ptheta0:.5g}, psip0 = {self.psip0:.5g}, rho0 = {self.rho0:.5g}, mu = {self.mu:.5g}"
            )
            logger.debug(
                f"\tTime span (t0, tf, steps): ({self.t_eval[0]}, {self.t_eval[-1]}, {len(self.t_eval)})"
            )
            logger.info("--> Initial conditions setup successful.")

        def setup_logic_flags():
            """Sets up logic flags and initializes variables that must have an initial value"""

            logger.info("Setting up logic flags...")

            self.calculated_conversion_factors = False
            self.calculated_energies = False
            self.calculated_orbit_type = False
            self.calculated_orbit = False
            self.t_or_p = "Unknown"
            self.l_or_c = "Unknown"
            self.percentage_calculated = 0

            # Stored initially to avoid attribute errors
            self.z_0freq = self.z_freq = self.theta_0freq = self.theta_freq = None

            logger.info("--> Logic flags setup successful.")

        grab_configuration()
        setup_tokamak()
        setup_constants()
        setup_solver()
        setup_init_cond()
        setup_logic_flags()

        logger.info("--------Particle Initialization Completed--------\n")

    def __str__(self):
        info_str = (
            "Constants of motion:\n"
            + "\tParticle Energy (normalized):\tE = {:e}\n".format(self.E)
            + "\tParticle Energy (eV):\t\tE = {:e} eV\n".format(self.E_eV)
            + "\tParticle Energy (J):\t\tE = {:e} J\n".format(self.E_J)
            + f"\tToroidal Momenta:\t\tPζ = {self.Pzeta0}\n\n"
            + "Other Quantities:\n"
            + f'\tParticle of Species:\t\t"{self.species}"\n'
            + f"\tOrbit Type:\t\t\t{self.orbit_type_str}\n"
            + f"\tMajor Radius:\t\t\tR = {self.R} meters\n"
            + f"\tMinor Radius:\t\t\tα = {self.a} meters\n"
            + "\tToroidal Flux at wall:\t\tψ = {:n}\n".format(self.psi_wall)
            + "\tTime unit:\t\t\tω = {:e} Hz \n".format(self.w0)
            + "\tEnergy unit:\t\t\tE = {:e} J \n\n".format(self.E_unit)
            + self.solver_output
        )

        return info_str

    def run(self, info: bool = True, orbit=True, events: list = []):
        r"""Calculates the motion and attributes of the particle.

        This is the function that must be called after the initial conditions
        are set. It runs the needed methods in the correct order first and
        initializes the "Plot" class afterwards, which contains all the plotting
        methods.

        Args:
            info (bool, optional): Whether or not to print the particle's
                calculated attributes. Defaults to True.
        """
        logger.info("--------Particle's 'run' routine is called.---------")

        self._conversion_factors()
        self._energies()
        self._orbit_type()

        if orbit:
            start = time()
            solution = self._orbit(events)
            end = time()

            self.theta = solution["theta"]
            self.psi = solution["psi"]
            self.zeta = solution["zeta"]
            self.rho = solution["rho"]
            self.psip = solution["psip"]
            self.Ptheta = solution["Ptheta"]
            self.Pzeta = solution["Pzeta"]
            self.t_eval = solution["t_eval"]
            self.t_events = solution["t_events"]
            self.y_events = solution["y_events"]
            self.message = solution["message"]

            duration = f"{end-start:.4f}"
            self.calculated_orbit = True
            self.solver_output = (
                f"Solver output: {self.message}\n" + f"Orbit calculation time: {duration}s."
            )
            logger.info(f"Orbit calculation completed. Took {duration}s")
        else:
            self.solver_output = ""
            logger.info("\tOrbit calculation deliberately skipped.")

        if info:
            logger.info("Printing Particle.__str__() to stdout.")
            print(self.__str__())
        logger.info("Printing Particle.__str__():\n\t\t\t" + self.__str__())

        logger.info("Initializing composite class 'Plot'...")
        self.plot = Plot(self)
        logger.info("Composite class 'Plot' successfully initialized.")
        logger.info("---------Particle's 'run' routine completed--------\n")

    def _conversion_factors(self):
        r"""Calculates the conversion coeffecient needed to convert from lab to NU
        and vice versa."""
        logger.info("Calculating conversion factors...")

        e = self.e  # 1.6*10**(-19)C
        Z = self.zeta
        m = self.mass_kg  # kg
        B = self.Bfield.B0  # Tesla
        R = self.R  # meters

        self.w0 = abs(Z) * e * B / m  # [s^-1]
        self.E_unit = m * self.w0**2 * R**2  # [J]

        # Conversion Factors
        self.NU_to_eV = 1 / self.E_unit
        self.NU_to_J = e / self.E_unit
        self.Volts_to_NU = self.sign * self.E_unit

        self.calculated_conversion_factors = True
        logger.info("--> Calculated conversion factors.")

    def _energies(self):
        r"""Calculates the particle's energy in [NU], [eV] and [J], using
        its initial conditions.
        """
        r0 = sqrt(2 * self.psi0)
        B_init = self.Bfield.B(r0, self.theta0)
        Phi_init = float(self.Efield.Phi_of_psi(self.psi0))
        Phi_init_NU = Phi_init * self.Volts_to_NU

        self.E = (  # Normalized Energy from initial conditions
            (self.Pzeta0 + self.psip0) ** 2 * B_init**2 / (2 * self.Bfield.g**2 * self.mass_amu)
            + self.mu * B_init
            + self.sign * Phi_init_NU
        )

        self.E_eV = self.E * self.NU_to_eV
        self.E_J = self.E * self.NU_to_J

        self.calculated_energies = True
        logger.info("Calculated particle's energies(NU, eV, J).")

    def _orbit_type(self):
        r"""
        Estimates the orbit type given the initial conditions ONLY.

        .. caution:: This method works only in the absence of an Electric Field.

        Trapped/passing:
        The particle is trapped if rho vanishes, so we can
        check if rho changes sign. Since
        :math:`\rho = \dfrac{\sqrt{2W-2\mu B}}{B}`, we need only to
        check under the root.

        Confined/lost:
        (from shape page 87)
        We only have to check if the particle is in-between the 2 left parabolas.
        """
        logger.info("Calculating particle's orbit type:")

        if (self.has_efield) or (not self.Bfield.is_lar):
            self.orbit_type_str = (
                "Cannot calculate (Electric field is present, or Magnetic field is not LAR.)"
            )
            logger.warning(
                "\tElectric field is present, or Magnetic field is not LAR. Orbit type calculation is skipped."
            )
            return

        # Calculate Bmin and Bmax. In LAR, B decreases outwards.
        Bmin = self.Bfield.B(self.r_wall, 0)  # "Bmin occurs at psi_wall, θ = 0"
        Bmax = self.Bfield.B(self.r_wall, np.pi)  # "Bmax occurs at psi_wall, θ = π"

        # Find if trapped or passing from rho (White page 83)
        sqrt1 = 2 * self.E - 2 * self.mu * Bmin
        sqrt2 = 2 * self.E - 2 * self.mu * Bmax
        if sqrt1 * sqrt2 < 0:
            self.t_or_p = "Trapped"
        else:
            self.t_or_p = "Passing"
        logger.debug(f"\tParticle found to be {self.t_or_p}.")

        # Find if lost or confined
        self.orbit_x = self.Pzeta0 / self.psip0
        self.orbit_y = self.mu / self.E
        logger.debug("\tCallling Construct class...")
        foo = Construct(self, get_abcs=True)

        # Recalculate y by reconstructing the parabola (there might be a better way
        # to do this)
        upper_y = foo.abcs[0][0] * self.orbit_x**2 + foo.abcs[0][1] * self.orbit_x + foo.abcs[0][2]
        lower_y = foo.abcs[1][0] * self.orbit_x**2 + foo.abcs[1][1] * self.orbit_x + foo.abcs[1][2]

        if self.orbit_y < upper_y and self.orbit_y > lower_y:
            self.l_or_c = "Confined"
        else:
            self.l_or_c = "Lost"
        logger.debug(f"\tParticle found to be {self.l_or_c}.")

        self.orbit_type_str = self.t_or_p + "-" + self.l_or_c

        self.calculated_orbit_type = True
        logger.info(f"--> Orbit type completed. Result: {self.orbit_type_str}.")

    def _orbit(self, events: list = []):

        t = self.t_eval

        init_cond = {
            "theta0": self.theta0,
            "psi0": self.psi0,
            "zeta0": self.zeta0,
            "rho0": self.rho0,
        }

        constants = {
            "E": self.E,
            "mu": self.mu,
            "Pzeta0": self.Pzeta0,
        }

        profile = {
            "q": self.q,
            "Bfield": self.Bfield,
            "Efield": self.Efield,
            "Volts_to_NU": self.Volts_to_NU,
        }

        return orbit(t, init_cond, constants, profile, events)

    def freq_analysis(self):

        pass
