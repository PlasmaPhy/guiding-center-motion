"""
This module initializes the "Particle" class, which calculates the orbit,
orbit type, and can draw several different plots
"""

import numpy as np
from time import time
from scipy.integrate import odeint, solve_ivp
from math import sqrt, sin, cos
from .plot import Plot
from .parabolas import Construct
from .freq import FreqAnalysis
from .bfield import MagneticField
from .efield import ElectricField, Nofield
from .qfactor import QFactor
from . import config, logger


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
        method: str = "RK45",
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
            method (str): Method used by the solver. Can be either 'lsoda' or 'RK45'.
                Defaults to 'RK45'.
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

            if method in ["RK45", "lsoda"] and isinstance(rtol, (int, float)):
                self.method = method
                self.rtol = rtol
            else:
                logger.warning("Invalid passed solver method. Using defaults...")
                self.method = self.configs["default_method"]
                self.rtol = float(self.configs["rtol"])  # Only used in RK45

            logger.debug(
                f"\tUsing solver method '{self.method}', with relative tolerance of {self.rtol} (only used by RK45)."
            )
            logger.info("--> Solver setup successful.")

        def setup_init_cond():
            """Sets up the particles initial condition and parameters, as well as the solver's S0."""

            logger.info("Setting up particle's initial conditions...")

            self.mu = mu
            self.theta0 = init_cond[0]
            init_cond[1] *= self.psi_wall  # CAUTION! Normalize it to psi_wall
            self.psi0 = init_cond[1]
            self.z0 = init_cond[2]
            self.Pz0 = init_cond[3]
            self.psip0 = q.psip_of_psi(self.psi0)
            self.t_eval = t_eval
            self.t_eval_given = self.t_eval.copy()  # When re-running _orbit
            self.rho0 = self.Pz0 + self.psip0  # Pz0 + psip0
            self.ode_init = [self.theta0, self.psi0, self.z0, self.rho0]

            formatted_ode_init = [float(f"{_:.5g}") for _ in self.ode_init]
            ode_init_dict = dict(zip(["theta0", "psi0", "z0", "rho0"], formatted_ode_init))

            logger.debug(f"\tSolver initial conditions: {ode_init_dict}.")
            logger.debug(f"\tOther initial conditions: Pz0 = {self.Pz0}, psip0 = {self.psip0}")
            logger.debug(
                f"\tParameters: μ = {self.mu:.2e}, teval(t0, tf, steps) = ({self.t_eval[0]}, {self.t_eval[-1]}, {len(self.t_eval)})"
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
            + "\tParticle Energy (normalized):\tE  = {:e}\n".format(self.E)
            + "\tParticle Energy (eV):\t\tE  = {:e} eV\n".format(self.E_eV)
            + "\tParticle Energy (J):\t\tE  = {:e} J\n".format(self.E_J)
            + f"\tToroidal Momenta:\t\tPζ = {self.Pz0}\n\n"
            + "Other Quantities:\n"
            + f'\tParticle of Species:\t\t"{self.species}"\n'
            + f"\tOrbit Type:\t\t\t{self.orbit_type_str}\n"
            + f"\tMajor Radius:\t\t\tR = {self.R} meters\n"
            + f"\tMinor Radius:\t\t\tα = {self.a} meters\n"
            + "\tToroidal Flux at wall:\t\tψ = {:n}\n".format(self.psi_wall)
            + "\tTime unit:\t\t\tω = {:e} Hz \n".format(self.w0)
            + "\tEnergy unit:\t\t\tE = {:e} J \n\n".format(self.E_unit)
            + self.time_str
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

            (
                self.theta,
                self.psi,
                self.psip,
                self.zeta,
                self.rho,
                self.Ptheta,
                self.Pzeta,
                self.t_events,
                self.t_eval,
            ) = self._orbit(events=events)

            end = time()
            duration = f"{end-start:.4f}"
            self.calculated_orbit = True
            self.time_str = f"Orbit calculation time: {duration}s."
            logger.info(f"Orbit calculation completed. Took {duration}s")
        else:
            self.time_str = ""
            logger.info("\tOrbit calculation deliberately skipped.")

        if info:
            logger.info("Printing Particle.__str__().")
            print(self.__str__())

        logger.info("Initializing composite class 'Plot'...")
        self.plot = Plot(self)
        logger.info("Composite class 'Plot' successfully initialized.")
        logger.info("---------Particle's 'run' routine completed--------\n")

    def _orbit(self, events: list = [], t_eval=None):
        r"""Calculates the orbit of the particle, as well as
        :math:`P_\theta` and :math:`\psi_p`.

        Calculates the time evolution of the dynamical variables
        :math:`\theta, \psi, \zeta, \rho_{||}`. Afterwards, it calculates
        the canonical momenta :math:`P_\theta` and :math:`P_\zeta`, and the
        poloidal flux :math:`\psi_p` through the q factor.

        The orbit can be calculated with SciPy's ``solve_ivp`` or ``odeint``
        methods. The default is ``solve_ivp`` (RK4(5)).

        The ``solve_ivp`` function integrates an ODE system using a 4th
        order Runge-Kutta method, with an error estimation of 5th order
        (RK4(5)). The default relative tolerance is :math:`10^{-6}`.

        The ``odeint`` function integrates an ODE system using lsoda
        from the FORTRAN library odepack.

        Orbit is stored in "self".
        """
        logger.info(f"Calculating orbit with events {events}")

        if t_eval is None:
            t_eval = self.t_eval_given
            logger.debug("\tUsing given teval.")

        def dSdt(t, S):
            """Sets the diff equations system to pass to scipy.

            All values are in normalized units (NU).
            """

            theta, psi, z, rho = S

            # Intermediate values
            phi_der_psip, phi_der_theta = self.Efield.Phi_der(psi)
            phi_der_psip *= self.Volts_to_NU
            phi_der_theta *= self.Volts_to_NU
            q_value = self.q.q_of_psi(psi)
            sin_theta = sin(theta)
            cos_theta = cos(theta)
            r = sqrt(2 * psi)
            B = self.Bfield.B(r, theta)
            par = self.mu + rho**2 * B
            bracket1 = -par * q_value * cos_theta / r + phi_der_psip
            bracket2 = par * r * sin_theta + phi_der_theta
            D = self.Bfield.g * q_value + self.Bfield.I

            # Canonical Equations
            theta_dot = 1 / D * rho * B**2 + self.Bfield.g / D * bracket1
            psi_dot = -self.Bfield.g / D * bracket2 * q_value
            rho_dot = psi_dot / (self.Bfield.g * q_value)
            z_dot = rho * B**2 / D - self.Bfield.I / D * bracket1

            return [theta_dot, psi_dot, z_dot, rho_dot]

        if self.method == "RK45":
            t_span = (t_eval[0], t_eval[-1])
            sol = solve_ivp(
                dSdt,
                t_span=t_span,
                y0=self.ode_init,
                t_eval=t_eval,
                rtol=self.rtol,
                events=events,
            )
            theta = sol.y[0]
            psi = sol.y[1]
            z = sol.y[2]
            rho = sol.y[3]
            t_events = sol.t_events
            t_eval = sol.t
        elif self.method == "lsoda":
            sol = odeint(dSdt, y0=self.ode_init, t=self.t_eval, tfirst=True)
            theta = sol.T[0]
            psi = sol.T[1]
            z = sol.T[2]
            rho = sol.T[3]
        else:
            print("Solver method must be either 'lsoda' or 'RK45'.")

        # Calculate psip and Canonical Momenta
        psip = self.q.psip_of_psi(psi)
        Ptheta = psi + rho * self.Bfield.I
        Pzeta = rho * self.Bfield.g - psip

        return [theta, psi, psip, z, rho, Ptheta, Pzeta, t_events, t_eval]

    def events(self, key):

        def single_theta_period(t, S):
            return (S[0] - self.theta0) or (S[1] - self.psi0)

        single_theta_period.terminal = 2
        single_theta_period.direction = 1
        events_dict = {"single_theta_period": single_theta_period}

        def event2(t, S):
            pass

        return events_dict[key]

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
            (self.Pz0 + self.psip0) ** 2 * B_init**2 / (2 * self.Bfield.g**2 * self.mass_amu)
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
        self.orbit_x = self.Pz0 / self.psip0
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

    def afreq_analysis(
        self,
        angle: str,
        trim: bool = True,
        normal: bool = False,
        remove_bias: bool = True,
        info: bool = True,
    ):
        r"""Given the input angle, runs Frequncy Analysis and prints results.

        Trimming and removing the bias of the timeseries gives the best results, both in
        peak-to-peak calculation of the frequency and FFT.

        Calling this functions also stores the 0th and 1st frequency of the angle, which are
        needed to calculate the q_kinetic.

        Args:
            angle (str): The angle to analyse,
            trim (bool, optional): Whether or not to trim the edges of the time series.
                Defaults to True.
            normal (bool, optional): Whether or not to use normal or lab units.
                Defaults to True.
            remove_bias (bool, optional): Whether or not to remove the signal's bias.
                Defaults to True.
            info (bool, optional): Whether or not to print results. Defaults to True.
        Returns:
            str: The results of the analysis.
        """

        if not remove_bias:
            print("Warning: not removing the biases returns nonsense frequencies!")

        def run_fourier():
            """Necessary steps to run the analysis.

            Returns:
                tuple: the 0-th and first frequency of the signal.
            """

            self.FreqAnalysis = FreqAnalysis(
                self, x, angle, trim=trim, normal=normal, remove_bias=remove_bias
            )

            # Plot Object needs to be re-initialized
            self.plot = Plot(self)

            # Check if the frequencies are correctly calculated
            if self.FreqAnalysis.q_kinetic_ready:
                return self.FreqAnalysis.get_omegas()
            else:
                return (None, None)

        # Also Inintialize frequencies to make sure they are calculated later
        if angle == "theta":
            x = self.theta.copy()
            self.theta_0freq = self.theta_freq = None
            self.theta_0freq, self.theta_freq = run_fourier()
        elif angle == "zeta":
            x = self.zeta.copy()
            self.z_0freq = self.z_freq = None
            self.z_0freq, self.z_freq = run_fourier()

        # Print results
        if info and self.FreqAnalysis.signal_ok:
            return print(self.FreqAnalysis)

    def freq_analysis(self, angle: str, sine: bool = False, trim_params: dict = {}):

        obj = FreqAnalysis(self, angle, sine=sine, trim_params=trim_params)
        obj.run()
        print(obj)

        # Plot Object needs to be re-initialized
        self.plot = Plot(self)
        self.plot._fft(obj)
