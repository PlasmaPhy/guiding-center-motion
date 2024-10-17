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

tokamak_profile = {

    "contour_params": {
        "levels": 100,
        "cmap": "winter",
    },
}

drift = {

    "scatter_args": {
        "s" : 0.1,
        "color" : "red",
    },

    "yfontsize": 20,
    "xfontsize": 20,
}

contour_energy = {

    "contour_grid_density" : 100,
    "contour_levels" : 15,
    "contour_cmap" : "plasma",
    "cbar_color": "k",
}

parabolas = {
    
    "parabolas_normal_kw": {
        "color" : "b",
        "linewidth" : 0.6,
    },

    "parabolas_dashed_kw": {
        "color" : "b",
        "linewidth" : 0.6,
    },
}

orbit_point = {

    "orbit_point_kw": {
        "s" : 15,
        "marker" : "o",
        "edgecolor" : "k",
        "facecolor" : "red",
    },
}

torus2d = {

    "wall_points": 2000,

    "torus2d_wall_kw": {
        "color" : "k",
        "s" : 0.5,
    },
    
    "torus2d_orbit_kw": {
        "color" : "blue",
        "s" : 0.07,
    },
}

torus3d = {
    # Higher rstride/cstride and lower points give better performance
    "rstride": 5,
    "cstride": 5,
    "wall_points": 301,

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
