import numpy as np
import matplotlib.pyplot as plt


class Construct:
    """Constructs the orbit type parabolas, as well as the
    trapped-passing boundary and plots them.
    """

    def __init__(self, cwp, get_abcs=False, limit_axis=True):
        r"""Copies attributes from cwp to self.

        The instance itself is initialized internally by the Plot class, and
        should not be called by the user.

        Args:
            cwp (Particle): The Current Working Particle
        """
        self.__dict__ = dict(cwp.__dict__)
        self.get_abcs = get_abcs
        self.limit_axis = limit_axis
        self._setup()

        if self.get_abcs:  # Just get the coefficients and return
            self.return_abcs()
            return

        self._plot_parabolas()
        self._plot_tp_boundary()

    def _setup(self):
        """Calculates the parabolas' constants, x-intercepts,
        maximums, and sets the x-limits
        """
        mu, psi_wall, g = self.mu, self.psi_wall, self.g

        Bmin = 1 - np.sqrt(2 * psi_wall)  # "Bmin occurs at psip_wall, θ = 0"
        Bmax = 1 + np.sqrt(2 * psi_wall)  # "Bmax occurs at psip_wall, θ = π"

        # Parabolas constants [a, b, c]
        # __________________________________________________________
        # Top left
        E1 = mu * Bmin
        abc1 = [
            -Bmin * psi_wall**2 / (2 * g**2 * E1),
            -Bmin * psi_wall**2 / (g**2 * E1),
            -Bmin * psi_wall**2 / (2 * g**2 * E1) + 1 / Bmin,
        ]
        # Bottom left
        E2 = mu * Bmax
        abc2 = [
            -Bmax * psi_wall**2 / (2 * g**2 * E2),
            -Bmax * psi_wall**2 / (g**2 * E2),
            -Bmax * psi_wall**2 / (2 * g**2 * E2) + 1 / Bmax,
        ]
        # Right (Magnetic Axis)
        E3 = mu
        abc3 = [
            -(psi_wall**2) / (2 * g**2 * E3),
            0,
            1,
        ]
        # __________________________________________________________

        # Calculate all x-intercepts and use the 2 outermost
        self.abcs = [abc1, abc2, abc3]

        if self.get_abcs:
            return

        x_intercepts = np.array(3)
        self.par1 = _Parabola(self.abcs[0])  # Top Left
        self.par2 = _Parabola(self.abcs[1])  # Bottom Left
        self.par3 = _Parabola(self.abcs[2])  # Right

        for par in [self.par1, self.par2, self.par3]:
            if par.discriminant <= 0:
                print("Parabola's discriminant is zero. Aborting...")
                return

        x_intercepts = np.array(
            [
                self.par1._get_x_intercepts(),
                self.par2._get_x_intercepts(),
                self.par3._get_x_intercepts(),
            ]
        )

        extremums = np.array(
            [
                self.par1._get_extremum()[1],
                self.par2._get_extremum()[1],
                self.par3._get_extremum()[1],
            ]
        )

        self.xlim = [1.02 * x_intercepts.min(), 1.02 * x_intercepts.max()]
        self.ylim = [0, 1.1 * extremums.max()]

    def _plot_parabolas(self):
        """Plots the 3 parabolas."""

        # Top left
        x, y = self.par1._construct(self.xlim)
        plt.plot(x, y, **self.Config.parabolas_normal_plot_kw)

        # Bottom left
        x, y = self.par2._construct(self.xlim)
        plt.plot(x, y, linestyle="--", **self.Config.parabolas_dashed_plot_kw)

        # Right
        x, y = self.par3._construct(self.xlim)
        plt.plot(x, y, linestyle="dashdot", **self.Config.parabolas_dashed_plot_kw)

        # General plot settings
        top_par = _Parabola(self.abcs[0])
        _, top = top_par._get_extremum()
        plt.gca().set_ylim(bottom=self.ylim[0])
        if self.limit_axis:
            plt.gca().set_xlim(self.xlim)
            plt.gca().set_ylim(top=self.ylim[1])
        plt.ylabel(r"$\dfrac{\mu B_0}{E}$", rotation=0)
        plt.xlabel(r"$P_\zeta/\psi_p$")
        plt.title(r"Orbit types in the plane of $P_\zeta - \mu$ for fixed energy.", c="b")

    def _plot_tp_boundary(self):
        """Plots the Trapped-Passing Boundary."""

        # Vertical line
        foo = _Parabola(self.abcs[0])
        p1 = foo._get_extremum()
        foo = _Parabola(self.abcs[1])
        p2 = foo._get_extremum()

        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], **self.Config.vertical_line_plot_kw)

        # Sideways parabola
        x1 = self.par2._get_extremum()[0]
        x2 = self.par3._get_extremum()[0]

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

    def return_abcs(self):
        """Returns the consants of the 3 parabolas as [[...],[...],[...]]
        Used in determining particle's orbit type.
        """
        return self.abcs


# ______________________
class _Parabola:
    """Creates a general-form parabola :math:`ax^2 + bx + c = 0`,
    calculates intercepts and extremums.

    Should only be used internally by the class ``Construct``

    Both x and y are normalized, :math:`x = Pz/psip_wall` and :math:`y=μB0/E`.
    """

    def __init__(self, abc: np.array):
        """Initialization and intercepts/extremums calculation.

        Args:
            abc (np.array): 1x3 array containing the 3 constants.
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

    def _get_x_intercepts(self):
        """Returns the 2 x-intercepts as an array.

        Returns:
            np.array: 1D np.array containing the 2 x-intercepts.
        """
        return self.x_intercepts

    def _get_extremum(self):
        """Returns the extremum point as (x,y)

        Returns:
            2-tuple: The extremum point
        """
        if self.a > 0:
            return [self.min_pos, self.min]
        else:
            return [self.max_pos, self.max]

    def _construct(self, xlim):
        """Constructs x and y arrays, ready to be plotted.

        Args:
            xlim (list): The x interval. Determined by the plotter
                    so that all 3 parabolas are constructed at the
                    same interval.

        Returns:
            2-tuple of 1D np.arrays: The x and y to be plotted.
        """

        x = np.linspace(xlim[0], xlim[1], 1000)
        y = self.a * x**2 + self.b * x + self.c

        return (x, y)
