"""
This class constructs the parabolas that classify each orbit type
"""
import numpy as np
import json
import matplotlib.pyplot as plt

class Orbit_parabolas:
    """
    Constructs a parabola ax^2 + bx + c = 0, where both x and y are normalized.
    Specifically, x = Pz/psi_wall and y=μB0/E
    """
    def __init__(self, E, g=1, psi_wall = 1):
        self.E = E
        self.psi_wall = psi_wall

        Bmin = 1 - np.sqrt(2*psi_wall) # "Bmin occurs at psi_wall, θ = 0"
        Bmax = 1 + np.sqrt(2*psi_wall) # "Bmax occurs at psi_wall, θ = π"
        B0 = 1

        # Parabolas constants [a, b, c]
        #__________________________________________________________
        # Top left
        abc1 = [-B0*Bmin*psi_wall**2/(2*g**2*E),
                -B0*Bmin*psi_wall**2/(g**2*E),
                -B0*Bmin*psi_wall**2/(2*g**2*E) + B0/Bmin]
        # Bottom left
        abc2 = [-B0*Bmax*psi_wall**2/(2*g**2*E),
                -B0*Bmax*psi_wall**2/(g**2*E),
                -B0*Bmax*psi_wall**2/(2*g**2*E) + B0/Bmax]
        # Right
        abc3 = [-B0**2*psi_wall**2/(2*g**2*E),
                0,
                1]
        #__________________________________________________________
        
        #Calculate all x-intercepts and use the 2 outermost
        self.abcs = [abc1, abc2, abc3]

        x_intercepts = np.array([])
        self.par1 = Parabola(self.abcs[0]) # Top Left
        self.par2 = Parabola(self.abcs[1]) # Bottom Left
        self.par3 = Parabola(self.abcs[2]) # Right

        x_intercepts = np.array([self.par1.get_x_intercepts(), self.par2.get_x_intercepts(), self.par3.get_x_intercepts()])
        
        self.xlim = [x_intercepts.min(), x_intercepts.max()]

        # Grab configuration
        with open("config.json") as jsonfile:
            self.config = json.load(jsonfile)

    def plot(self):
        
        normal_plot_kw = {
            "color": self.config["parabolas_color"],
            "linewidth": self.config["parabolas_width"]
        }
        dashed_plot_kw = {
            "color": self.config["dashed_color"],
            "linewidth": self.config["dashed_width"],
        }
        
        # Top left
        x, y = self.par1.construct(self.xlim)
        plt.plot(x, y, **normal_plot_kw)

        # Bottom left
        x, y = self.par2.construct(self.xlim)
        plt.plot(x, y, linestyle = "--", **dashed_plot_kw)

        # Right
        x, y = self.par3.construct(self.xlim)
        plt.plot(x, y, linestyle = "dashdot", **dashed_plot_kw)

        # General plot settings
        plt.gca().set_xlim(self.xlim)
        plt.gca().set_ylim(bottom = 0)
        plt.xlabel("${P_\zeta}/{\psi_{wall}}$")
        plt.ylabel("$\dfrac{\mu B_0}{E}\t$", rotation = 0)
        plt.title("Orbit types in the plane of $P_\zeta - \mu$ for fixed energy.")

        return

    def plot_vertical(self):
        foo = Parabola(self.abcs[0])
        p1 = foo.get_extremum()
        foo = Parabola(self.abcs[1])
        p2 = foo.get_extremum()

        plot_kw = {
            "color": self.config["dashed_color"],
            "linewidth": self.config["dashed_width"],
            "linestyle": "--"
        }
        
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], **plot_kw)

        return

    def plot_tp_boundary(self):
        x1 = self.par2.get_extremum()[0]
        x2 = self.par3.get_extremum()[0]

        x = np.linspace(x1, x2, 100)

        B0 = 1
        B1 = 1 - np.sqrt(-2*self.psi_wall*x)
        B2 = 1 + np.sqrt(-2*self.psi_wall*x)
        
        dashed_plot_kw = {
            "color": self.config["dashed_color"],
            "linewidth": self.config["dashed_width"],
            "linestyle": "--"
        }
        y1_plot = B0/B1
        y2_plot = B0/B2

        plt.plot(x, y1_plot, **dashed_plot_kw)
        plt.plot(x, y2_plot, **dashed_plot_kw)

        return

    def get_abcs(self):
        return(self.abcs)

class Parabola:
    """Creates a parabola ax^2 + bx + c = 0, where both x and y are normalized.
    Specifically, x = Pz/psi_wall and y=μB0/E.
    Stores minimum/maximum and intercepts

    """
    
    def __init__(self, abc):
        self.a = abc[0]
        self.b = abc[1]
        self.c = abc[2]

        # Calculate intrecepts/ extremums:
        self.discriminant = self.b**2 - 4*self.a*self.c
        self.x_intercepts = np.array([-self.b - np.sqrt(self.discriminant),
                                      -self.b + np.sqrt(self.discriminant)])/(2*self.a)
        self.y_intercept = self.c

        if self.a > 0:
            self.min_pos = - self.b/(2*self.a)
            self.min = self.a*self.min_pos**2 + self.b*self.min_pos + self.c
            self.max_pos = "Not defined"
            self.max = "Not Defined"
        else:
            self.min_pos = "Not Defined"
            self.min = "Not Defined"
            self.max_pos = - self.b/(2*self.a)
            self.max = self.a*self.max_pos**2 + self.b*self.max_pos + self.c
        
        # Grab configuration
        with open("config.json") as jsonfile:
            self.config = json.load(jsonfile)
    
    def get_x_intercepts(self):
        return self.x_intercepts

    def get_extremum(self):
        if self.a > 0:
            return [self.min_pos, self.min]
        else:
            return [self.max_pos, self.max]


    def construct(self, xlim):
        x = np.linspace(xlim[0], xlim[1], 1000)
        y = self.a*x**2 + self.b*x + self.c
        return [x, y]