import json

class Config_file():
    """Returns an object with plotting settings from config.json

    Plotting settings are stored as dictionaries stored in the object.
    """

    def __init__(self):

        # Set plot configurations dictionary from json file
        with open("config.json") as jsonfile:
            self.config = json.load(jsonfile)

        # Time plots
        self.time_scatter_kw = {
            "s": self.config["time_plots_size"],
            "color": self.config["time_plots_color"],
        }
        self.time_ylabel_kw = {
            "rotation": 0,
            "fontsize": self.config["time_plots_ylabel_fontsize"],
        }

        # Drift plots
        self.drift_scatter_kw = {
            "s": self.config["drift_scatter_size"],
            "color": self.config["drift_scatter_color"],
        }
        self.drift_plot_kw = {
            "linewidth": self.config["drift_plots_width"],
            "color": self.config["drift_plots_color"],
        }
        self.drift_ylabel_kw = {
            "rotation": 0,
            "fontsize": self.config["drift_plots_ylabel_fontsize"],
        }
        self.drift_xlabel_kw = {
            "rotation": 0,
            "fontsize": self.config["drift_plots_xlabel_fontsize"]
        }

        # Orbit point
        self.orbit_point_kw = {
            "markersize": self.config["orbit_point_size"],
            "marker": self.config["orbit_point_marker"],
            "markeredgecolor": self.config["orbit_point_edge_color"],
            "markerfacecolor": self.config["orbit_point_face_color"]
        }

        # Tori
        self.torus2d_wall_kw = {
            "c": self.config["torus2d_wall_color"],
            "s": self.config["torus2d_wall_size"]
        }
        self.torus2d_orbit_kw = {
            "c": self.config["torus2d_orbit_color"],
            "s": self.config["torus2d_orbit_size"]
        }
        self.torus3d_wall_kw = {
            "color": self.config["torus3d_wall_color"],
            "alpha": self.config["torus3d_wall_alpha"],
        }
        self.torus3d_orbit_kw = {
            "color": self.config["torus3d_orbit_color"],
            "alpha": self.config["torus3d_orbit_alpha"],
            "linewidth": self.config["torus3d_orbit_size"]
        }

        # Parabolas
        self.parabolas_normal_plot_kw = {
            "color": self.config["parabolas_color"],
            "linewidth": self.config["parabolas_width"]
        }
        self.parabolas_dashed_plot_kw = {
            "color": self.config["dashed_color"],
            "linewidth": self.config["dashed_width"],
        }
        self.vertical_line_plot_kw = {
            "color": self.config["dashed_color"],
            "linewidth": self.config["dashed_width"],
            "linestyle": "--"           
        }

        self.contour_grid_density = self.config["contour_grid_density"]
        self.contour_levels_default = self.config["contour_levels_default"]
        self.contour_cmap = self.config["contour_cmap"]


    
    def __getitem__(self, kw):
        """Makes object subscriptable

        kw dictionaries should be called as "Config.<kw>"
        """
        return self.config.kw