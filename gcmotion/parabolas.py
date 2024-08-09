"""This class constructs the parabolas that classify each orbit type"""

import numpy as np
import matplotlib.pyplot as plt
from . import utils


class OrbitParabolas:
    """
    Constructs 3 parabolas  ax^2 + bx + c = 0, and finds the special points.

    Both x and y are normalized, x = Pz/psip_wall and y=μB0/E.
    to change the parabolas, go to __init__() and change the constants between
    the dashed comments.
    """

    def __init__(self, R, a, mu, Bfield, Efield, Volts_to_NU):
        """Constructs the 3 parabolas and calculates the special points.

        :param E: (float) The particle's energy
        :param B: (list) The 3 componets of the contravariant representation
            of the magnetic field B.
        :param psip_wall: (float) The value of ψ at the wall. Better be lower
            lower than 0.5.
        """
        # Grab configuration
        self.Config = utils.ConfigFile()

        self.R = R
        self.a = a
        self.mu = mu
        self.I, self.g, self.B0 = Bfield
        self.Efield = Efield
        self.r_wall = self.a / self.R
        self.psi_wall = (self.r_wall) ** 2 / 2  # normalized to R
        self.Volts_to_NU = Volts_to_NU

        self.B0 = 1
        self.Bmin = self.B0 * (1 - np.sqrt(2 * self.psi_wall))  # "Bmin occurs at psip_wall, θ = 0"
        self.Bmax = self.B0 * (1 + np.sqrt(2 * self.psi_wall))  # "Bmax occurs at psip_wall, θ = π"

        # Electric Potential Components:
        self.Phi0 = self.Efield.Phi_of_psi(0) * self.Volts_to_NU
        self.Phi_wall = self.Efield.Phi_of_psi(self.psi_wall) * self.Volts_to_NU
        m = 1
        # Parabolas constants [a, b, c]
        # __________________________________________________________
        # Top left
        self.E1 = mu * self.Bmin + self.Phi_wall
        abc1 = [
            -self.B0 * self.Bmin * self.psi_wall**2 / (2 * self.g**2 * self.E1 * m),
            -self.B0 * self.Bmin * self.psi_wall**2 / (self.g**2 * self.E1 * m),
            -self.B0 * self.Bmin * self.psi_wall**2 / (2 * self.g**2 * self.E1 * m)
            + self.B0 / self.Bmin,
        ]
        # Bottom left
        self.E2 = mu * self.Bmax + self.Phi_wall
        abc2 = [
            -self.B0 * self.Bmax * self.psi_wall**2 / (2 * self.g**2 * self.E2 * m),
            -self.B0 * self.Bmax * self.psi_wall**2 / (self.g**2 * self.E2 * m),
            -self.B0 * self.Bmax * self.psi_wall**2 / (2 * self.g**2 * self.E2 * m)
            + self.B0 / self.Bmax,
        ]
        # Right (Magnetic Axis)
        self.E3 = mu * self.B0 + self.Phi0
        abc3 = [
            -(self.B0**2) * self.psi_wall**2 / (2 * self.g**2 * self.E3 * m),
            0,
            1,
        ]
        # __________________________________________________________

        # Calculate all x-intercepts and use the 2 outermost
        self.abcs = [abc1, abc2, abc3]

        x_intercepts = np.array([])
        self.par1 = Parabola(self.abcs[0])  # Top Left
        self.par2 = Parabola(self.abcs[1])  # Bottom Left
        self.par3 = Parabola(self.abcs[2])  # Right

        for par in [self.par1, self.par2, self.par3]:
            if par.discriminant <= 0:
                print("Parabola's discriminant is zero. Aborting...")
                return

        x_intercepts = np.array(
            [
                self.par1.get_x_intercepts(),
                self.par2.get_x_intercepts(),
                self.par3.get_x_intercepts(),
            ]
        )

        extremums = np.array(
            [self.par1.get_extremum()[1], self.par2.get_extremum()[1], self.par3.get_extremum()[1]]
        )

        self.xlim = [x_intercepts.min(), x_intercepts.max()]
        self.ylim = [0, 1.1 * extremums.max()]

    def plot_parabolas(self):
        """Plots the 3 parabolas."""

        # Top left
        x, y = self.par1.construct(self.xlim)
        plt.plot(x, y, **self.Config.parabolas_normal_plot_kw)

        # Bottom left
        x, y = self.par2.construct(self.xlim)
        plt.plot(x, y, linestyle="--", **self.Config.parabolas_dashed_plot_kw)

        # Right
        x, y = self.par3.construct(self.xlim)
        plt.plot(x, y, linestyle="dashdot", **self.Config.parabolas_dashed_plot_kw)

        # General plot settings
        plt.gca().set_xlim(self.xlim)
        top_par = Parabola(self.abcs[0])
        _, top = top_par.get_extremum()
        plt.gca().set_ylim(bottom=self.ylim[0], top=self.ylim[1])
        plt.ylabel(r"$\dfrac{\mu B_0}{E}$", rotation=0)
        plt.xlabel(r"$P_\zeta/\psi_p$")
        plt.title(r"Orbit types in the plane of $P_\zeta - \mu$ for fixed energy.", c="b")

    def plot_tp_boundary(self):
        """Plots the Trapped-Passing Boundary."""

        # Vertical line
        foo = Parabola(self.abcs[0])
        p1 = foo.get_extremum()
        foo = Parabola(self.abcs[1])
        p2 = foo.get_extremum()

        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], **self.Config.vertical_line_plot_kw)

        # Sideways parabola
        x1 = self.par2.get_extremum()[0]
        x2 = self.par3.get_extremum()[0]

        # E = self.mu * self.B0 + self.Phi_wall

        x = np.linspace(x1, x2, 1000)

        B1 = 1 + np.sqrt(-2 * self.psi_wall * x)  # θ = 0
        B2 = 1 - np.sqrt(-2 * self.psi_wall * x)  # θ = π

        # E1 = self.mu * self.Bmax + self.Phi_wall  # θ = 0
        # E2 = self.mu * self.Bmax + self.Phi0  # θ = π

        y1_plot = 1 / B1  # upper
        y2_plot = 1 / B2  # lower

        plt.plot(x, y1_plot, **self.Config.parabolas_dashed_plot_kw)
        plt.plot(x, y2_plot, **self.Config.parabolas_dashed_plot_kw)

    def get_abcs(self):
        """Returns the consants of the 3 parabolas as [[...],[...],[...]]"""

        return self.abcs


class Parabola:
    """Creates a parabola :math:`ax^2 + bx + c = 0`,

    Both x and y are normalized, :math:`x = Pz/psip_wall` and :math:`y=μB0/E`.
    Stores minimum/maximum and intercepts
    """

    def __init__(self, abc: np.array):  # Ready to commit
        """Initialization and intercepts/extremums calculation.

        :param abc: 1d array containing the 3 constants.
        """
        self.a = abc[0]
        self.b = abc[1]
        self.c = abc[2]

        # Calculate intrecepts/ extremums:
        self.discriminant = self.b**2 - 4 * self.a * self.c
        if self.discriminant < 0:
            return

        self.x_intercepts = np.array(
            [-self.b - np.sqrt(self.discriminant), -self.b + np.sqrt(self.discriminant)]
        ) / (2 * self.a)
        self.y_intercept = self.c

        if self.a > 0:
            self.min_pos = -self.b / (2 * self.a)
            self.min = self.a * self.min_pos**2 + self.b * self.min_pos + self.c
            self.max_pos = "Not defined"
            self.max = "Not Defined"
        else:
            self.min_pos = "Not Defined"
            self.min = "Not Defined"
            self.max_pos = -self.b / (2 * self.a)
            self.max = self.a * self.max_pos**2 + self.b * self.max_pos + self.c

        # Grab configuration
        # self.Config = utils.ConfigFile()

    def get_x_intercepts(self):  # Should fix the case that no intercepts exist
        """Returns the 2 x-intercepts as an array"""
        return self.x_intercepts

    def get_extremum(self):
        """Returns the extremum point as (x,y)"""
        if self.a > 0:
            return [self.min_pos, self.min]
        else:
            return [self.max_pos, self.max]

    def construct(self, xlim):
        """Constructs x and y arrays, ready to be plotted.

        Args:
            xlim (list): The x interval. Determined by the plotter
                    so that all 3 parabolas are constructed at the
                    same interval.
        """

        x = np.linspace(xlim[0], xlim[1], 1000)
        y = self.a * x**2 + self.b * x + self.c

        return [x, y]
