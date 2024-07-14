"""
This class constructs the parabolas that classify each orbit type
"""

import numpy as np
import utils
import matplotlib.pyplot as plt
import Functions.Qfactors as Qfactors


class Orbit_parabolas:
    """
    Constructs 3 parabolas  ax^2 + bx + c = 0, and finds the special points.

    Both x and y are normalized, x = Pz/psip_wall and y=μB0/E.
    to change the parabolas, go to __init__() and change the constants between
    the dashed comments.
    """

    def __init__(self, E, q=Qfactors.Unity, B=[0, 1, 0], Efield=None, psi_wall=0.3):  # Ready to commit
        """Constructs the 3 parabolas and calculates the special points.

        Args:
            E (float): The particle's energy
            B (list, optional): The 3 componets of the contravariant representation
                                of the magnetic field B. Defaults to [0, 1, 0].
            psip_wall (float, optional): The value of ψ at the wall. Better be lower
                                lower than 0.5. Defaults to 0.3.
        """

        self.q = q
        self.E = E
        self.i, self.g, self.delta = B
        self.Efield = Efield
        self.psi_wall = psi_wall
        self.psip_wall = self.q.psip_from_psi(psi_wall)

        Bmin = 1 - np.sqrt(2 * self.psip_wall)  # "Bmin occurs at psip_wall, θ = 0"
        Bmax = 1 + np.sqrt(2 * self.psip_wall)  # "Bmax occurs at psip_wall, θ = π"
        B0 = 1

        # Electric Potential Components:
        # if self.Efield is None:
        #     Phimin = Phimax = Phi0 = 0
        # else:
        #     Phimin, Phimax = self.Efield.extremums()
        #     Phi0 = self.Efield.Phi_of_r(0)

        # Parabolas constants [a, b, c]
        # __________________________________________________________
        # Top left
        abc1 = [
            -B0 * Bmin * self.psip_wall**2 / (2 * self.g**2 * E),
            -B0 * Bmin * self.psip_wall**2 / (self.g**2 * E),
            -B0 * Bmin * self.psip_wall**2 / (2 * self.g**2 * E) + B0 / Bmin,
        ]
        # Bottom left
        abc2 = [
            -B0 * Bmax * self.psip_wall**2 / (2 * self.g**2 * E),
            -B0 * Bmax * self.psip_wall**2 / (self.g**2 * E),
            -B0 * Bmax * self.psip_wall**2 / (2 * self.g**2 * E) + B0 / Bmax,
        ]
        # Right
        abc3 = [-(B0**2) * self.psip_wall**2 / (2 * self.g**2 * E), 0, 1]
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

        self.xlim = [x_intercepts.min(), x_intercepts.max()]

        # Grab configuration
        self.Config = utils.Config_file()

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
        plt.gca().set_ylim(bottom=0, top=1.1 * top)
        plt.xlabel("${P_\zeta}/{\psi_{wall}}$")
        plt.ylabel("$\dfrac{\mu B_0}{E}\t$", rotation=0)
        plt.title("Orbit types in the plane of $P_\zeta - \mu$ for fixed energy.")

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

        x = np.linspace(x1, x2, 100)

        B0 = 1
        B1 = 1 - np.sqrt(-2 * self.psip_wall * x)
        B2 = 1 + np.sqrt(-2 * self.psip_wall * x)

        y1_plot = B0 / B1
        y2_plot = B0 / B2

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
        self.x_intercepts = np.array([-self.b - np.sqrt(self.discriminant), -self.b + np.sqrt(self.discriminant)]) / (2 * self.a)
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
        # self.Config = utils.Config_file()

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
