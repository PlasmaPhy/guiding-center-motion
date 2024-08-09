import numpy as np
import yaml
from . import Particle, utils
import matplotlib.pyplot as plt


class Particles:
    """Calculates multiple orbits and plots."""

    def __init__(self, path, store=True):

        # Grab configuration
        self.Config = utils.ConfigFile()

        self.path = path
        self.params = self.get_params()
        self.store = store
        self.num_of_particles = len(self.params)
        # self.create()

        # if one string param -> multiply it by number of particles

    def get_params(self):

        with open(self.path, "r") as file:
            config = yaml.safe_load(file)
            num = config["n"]  # get number of particles
            del config["n"]

        # List of keys that need to be evaluated
        eval_keys = ["q", "Efield", "mu", "tspan"]

        params = []  # list of dicts

        for n in range(num):
            param = {}
            for key in config.keys():
                param[key] = config[key]

                if len(config[key]) == 1:
                    param[key] = [config[key][0]] * num
                else:
                    param[key][n] = config[key][n]

                if key in eval_keys:
                    param[key] = [eval(item) for item in param[key]]

            params.append(param)

        return params

    def create(self):
        """Creates the particles.

        Particles are stored inside the class as attributes.
        """

        # Particles creation
        self.particles = []
        # Iteration through params list
        for param in self.params:
            psi_wall = param["psi_wall"]
            R = param["R"]
            q = param["q"]
            B = param["B"]
            Efield = param["Efield"]

            species = param["species"]
            mu = param["mu"]
            theta0 = param["theta0"]
            psi0 = param["psi0"]
            z0 = param["z0"]
            Pz0 = param["Pz0"]
            tspan = param["tspan"]

            for i in range(self.num_of_particles):
                # Set initial conditions for each particle
                init_cond = [theta0[i], psi0[i], z0[i], Pz0[i]]

                # Create and store each particle based on their initial conditions
                particle = Particle.Particle(
                    species[i], init_cond, mu[i], tspan[i], q[i], R[i], B[i], Efield[i], psi_wall[i]
                )
                self.particles.append(particle)

    def plot_Ptheta_drifts(self, theta_lim, contour_Phi=True, units="keV", levels=None):
        """Plots all particle's Pθ drifts in the same plot and makes the contour plots in the background
        - theta_lim: usually [-π, π]
        - contour_Phi: if True it can include the electric potential in the contours calculation
        - units: Set to "eV" or "keV" according to preference
        - levels: Variable for the contours of the hamiltonian. Default number is 15.
        """

        self.fig = plt.figure(figsize=(6, 4))
        self.ax = self.fig.add_subplot(111)

        for idx, p in enumerate(self.particles):
            p.orbit()
            self.theta = p.theta
            self.psi = p.psi
            self.Pzeta = p.Pzeta
            self.Ptheta = p.Ptheta
            p.plot_Ptheta_drift(theta_lim, self.ax)

        # After drifts are plotted, store ylim to be passed at contour plot
        self.psi_lim = np.array(plt.gca().get_ylim()) * p.psi_wall

        # Calculate Energy values
        grid_density = self.Config.contour_grid_density
        theta, psi = np.meshgrid(
            np.linspace(theta_lim[0], theta_lim[1], grid_density),
            np.linspace(self.psi_lim[0], self.psi_lim[1], grid_density),
        )
        values = p.calcW_grid(theta, psi, p.Pz0, contour_Phi, units)
        span = np.array([values.min(), values.max()])

        # Set the levels (default = 15)
        if levels is None:
            levels = self.Config.contour_levels_default
        contour_kw = {
            "vmin": span[0],
            "vmax": span[1],
            "levels": levels,
            "cmap": self.Config.contour_cmap,
            "zorder": 1,
        }
        # Set units
        if units == "normal":
            label = "E (normalized)"
        elif units == "eV":
            label = "E (eV)"
        elif units == "keV":
            label = "E (keV)"
        else:
            print('units must be either "normal", "eV" or "keV"')
            return

        # Contour plot
        C = self.ax.contourf(theta, psi / p.psi_wall, values, **contour_kw)
        self.ax.set_xlabel(r"$\theta$")
        self.ax.set_ylabel(r"$\psi/\psi_{wall}$", rotation=90)
        ticks = ["-2π", "-3π/2", "-π", "-π/2", "0", "π/2", "π", "3π/2", "2π"]
        plt.xticks(np.linspace(-2 * np.pi, 2 * np.pi, 9), ticks)
        self.ax.set(xlim=[theta_lim[0], theta_lim[1]], ylim=self.psi_lim / p.psi_wall)
        self.ax.set_facecolor("white")
        cbar = self.fig.colorbar(C, ax=self.ax, fraction=0.03, pad=0.2, label=label)
