import numpy as np
import yaml
import Source.Particle as Particle
import matplotlib.pyplot as plt


class Particles:
    """Calculates multiple orbits and plots."""

    def __init__(self, path, store=True):
        self.path = path
        self.params = self.get_params()
        self.store = store
        self.num_of_particles = len(self.params)

    def get_params(self):

        with open(self.path, "r") as file:
            config = yaml.safe_load(file)
            num = config["n"]  # get number of particles
            del config["n"]

        params = []  # list of dicts

        for n in range(num):
            param = {}
            for key in config.keys():
                param[key] = config[key]
                param[key] = config[key][n]
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
            init_cond = [theta0, psi0, z0, Pz0]

            particle = Particle.Particle(species, init_cond, mu, tspan, q, B, Efield, psi_wall)
            self.particles.append(particle)

    def plot_Ptheta_drifts(self, theta_lim):
        """Plots all particle's Pθ drifts in the same plot"""

        for p in self.particles:
            p.plot_Ptheta_drift(theta_lim)
            plt.show

        # After drifts are plotted, store ylim to be passed at contour plot
        self.psi_lim = plt.get_ylim()

    # def contour_plot()
