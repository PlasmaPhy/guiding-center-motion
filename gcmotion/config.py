# fmt: off
configs = {
    # Particle Properties
    "elementary_charge" : 1.602176634e-19,  # C

    "e_mass_amu" : 0.0005446623,  # Units of proton mass
    "e_mass_keV" : 510998.95069,  # eV/c^2
    "e_mass_kg" : 9.1093837139e-31,  # kg
    "e_Z" : -1,  # Charge

    "p_mass_amu" : 1,  # Units of proton mass
    "p_mass_keV" : 938272000,  # eV/c^2
    "p_mass_kg" : 1.67262192e-27,  # kg
    "p_Z" : 1,  # Charge

    # Solvers
    "default_method" : "RK45",  # RK45 or lsoda
    "rtol" : 10e-8,

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

    "animation_kw": {
        "torus_color" : "cyan",
        "vaxis_color" : "blue",
        "particle_color" : "red",
        "flux_surface_color" : "red",
        "flux_surface_opacity" : 0.3,
        "percentage": 100, 
        "truescale": True, 
        "min_step": 0.01, 
        "seconds": 60,
    },
}
