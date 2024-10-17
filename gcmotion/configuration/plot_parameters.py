# fmt: off

time_evolution = {

    "fig_parameters": {
        "figsize": (13, 9),
        "sharex": True,
    },

    "scatter_args": {
        "s" : 0.01,
        "color" : "blue",
    },
    
    "ylabel_args": {
        "rotation": 0,
        "fontsize" : 15,
    },
}

#++++++++++++++++++
plot_parameters = {
    # Figures
    "dpi": 300,

    # Plots
    "time_plots_kw": {
        "s" : 0.08,
        "color" : "blue",
    },

    "time_plots_ylabel_kw": {
        "rotation": 0,
        "fontsize" : 10,
    },

    "drift_plots_kw": {
        "s" : 0.5,
        "color" : "red",
    },
    
    "drift_plots_ylabel_fontsize" : 20,
    "drift_plots_xlabel_fontsize" : 20,


    "drift_scatter_kw": {
        "s" : 0.1,
        "color" : "red",
    },

    "contour_grid_density" : 100,
    "contour_levels_default" : 15,
    "contour_cmap" : "plasma",

    "parabolas_normal_kw": {
        "color" : "b",
        "linewidth" : 0.6,
    },

    "parabolas_dashed_kw": {
        "color" : "b",
        "linewidth" : 0.6,
    },

    "orbit_point_kw": {
        "s" : 15,
        "marker" : "o",
        "edgecolor" : "k",
        "facecolor" : "red",
    },

    "torus2d_wall_kw": {
        "color" : "k",
        "s" : 0.3,
    },
    
    "torus2d_orbit_kw": {
        "color" : "blue",
        "s" : 0.07,
    },

    "torus3d_wall_kw": {
        "color" : "cyan",
        "alpha" : 0.3,
    },

    "torus3d_orbit_kw": {
        "color" : "red",
        "alpha" : 0.6,
        "linewidth" : 0.2,
    },

}
