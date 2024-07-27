"""
This class constructs the parabolas that classify each orbit type
"""

import numpy as np
import matplotlib.pyplot as plt
import Source.utils as utils


class Orbit_parabolas:
    """
    Constructs 3 parabolas  ax^2 + bx + c = 0, and finds the special points.

    Both x and y are normalized, x = Pz/psip_wall and y=μB0/E.
    to change the parabolas, go to __init__() and change the constants between
    the dashed comments.
    """

    def __init__(self, cwp):  # Ready to commit
        """Constructs the 3 parabolas and calculates the special points.

        Args:
            E (float): The particle's energy
            B (list, optional): The 3 componets of the contravariant representation
                                of the magnetic field B.
            psip_wall (float, optional): The value of ψ at the wall. Better be lower
                                lower than 0.5.
        """
        # Grab configuration
        self.Config = utils.ConfigFile()

        self.q = cwp.q
        self.E = cwp.E
        self.i, self.g, self.delta = cwp.B
        self.Efield = cwp.Efield
        self.psi_wall = cwp.psi_wall
        self.psip_wall = self.q.psip_of_psi(cwp.psi_wall)
        # self.psi_wall = self.psip_wall  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        B0 = 1  # cwp.B0
        Bmin = B0 * (1 - np.sqrt(2 * self.psi_wall))  # "Bmin occurs at psip_wall, θ = 0"
        Bmax = B0 * (1 + np.sqrt(2 * self.psi_wall))  # "Bmax occurs at psip_wall, θ = π"

        # Electric Potential Components:
        Phi0 = self.Efield.Phi_of_psi(0)
        Phi_wall = self.Efield.Phi_of_psi(self.psi_wall)

        # Parabolas constants [a, b, c]
        # __________________________________________________________
        # Top left
        abc1 = [
            -B0 * Bmin * self.psi_wall**2 / (2 * self.g**2 * self.E),
            -B0 * Bmin * self.psi_wall**2 / (self.g**2 * self.E),
            -B0 * Bmin * self.psi_wall**2 / (2 * self.g**2 * self.E)
            + B0 / Bmin * (1 - Phi_wall / self.E),
        ]
        # Bottom left
        abc2 = [
            -B0 * Bmax * self.psi_wall**2 / (2 * self.g**2 * self.E),
            -B0 * Bmax * self.psi_wall**2 / (self.g**2 * self.E),
            -B0 * Bmax * self.psi_wall**2 / (2 * self.g**2 * self.E)
            + B0 / Bmax * (1 - Phi_wall / self.E),
        ]
        # Right (Magnetic Axis)
        abc3 = [-(B0**2) * self.psi_wall**2 / (2 * self.g**2 * self.E), 0, -Phi0 / self.E + 1]
        # __________________________________________________________

        # Calculate all x-intercepts and use the 2 outermost
        self.abcs = [abc1, abc2, abc3]

        x_intercepts = np.array([])
        self.par1 = Parabola(self.abcs[0])  # Top Left
        self.par2 = Parabola(self.abcs[1])  # Bottom Left
        self.par3 = Parabola(self.abcs[2])  # Right

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

        # Grab configuration
        self.Config = utils.ConfigFile()

    def plot_parabolas(self):  # Ready to commit
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
        plt.ylabel("$\dfrac{\mu B_0}{E}\t$", rotation=0)
        plt.title("Orbit types in the plane of $P_\zeta - \mu$ for fixed energy.", c="b")
        plt.legend([f"Particle energy = {np.around(self.E,9)}"], loc="upper right", labelcolor="b")

    def plot_tp_boundary(self):  # Ready to commit
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

        x = np.linspace(x1, x2, 1000)

        B0 = 1
        B1 = 1 - np.sqrt(-2 * self.psi_wall * x)
        B2 = 1 + np.sqrt(-2 * self.psi_wall * x)

        y1_plot = B0 / B1 * (1 - self.Efield.Phi_of_psi(0) / self.E)  # upper
        y2_plot = B0 / B2 * (1 - self.Efield.Phi_of_psi(0) / self.E)  # lower

        plt.plot(x, y1_plot, **self.Config.parabolas_dashed_plot_kw)
        plt.plot(x, y2_plot, **self.Config.parabolas_dashed_plot_kw)

    def get_abcs(self):  # Ready to commit
        """Returns the consants of the 3 parabolas as [[...],[...],[...]]"""
        return self.abcs


class Parabola:
    """Creates a parabola ax^2 + bx + c = 0,

    Both x and y are normalized, x = Pz/psip_wall and y=μB0/E.
    Stores minimum/maximum and intercepts
    """

    def __init__(self, abc):  # Ready to commit
        """Initialization and intercepts/extremums calculation.

        Args:
            abc (array): 1d array containing the 3 constants.
        """
        self.a = abc[0]
        self.b = abc[1]
        self.c = abc[2]

        # Calculate intrecepts/ extremums:
        self.discriminant = self.b**2 - 4 * self.a * self.c
        if self.discriminant < 0:
            print("Parabola's discriminant is negative. Aborting...")
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
