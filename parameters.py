import gcmotion as gcm
import numpy as np

# To be passed to objects as parameters for ease
R = 6.2
a = 2
q = gcm.qfactor.Hypergeometric(R, a)
N = 15  # Number of particles

# fmt: off

params = {

    "R"           :   6.2,
    "a"           :   2,
    "q"           :   gcm.qfactor.Hypergeometric(R, a),
    "Bfield"      :   gcm.bfield.LAR(i = 0, g = 1, B0 = 5),
    "Efield"      :   None,#gcm.efield.Radial(R, a, q, Ea = 75000, minimum = 0.95, waist_width=20),
    "species"     :   "p",
    "mu"          :   10e-5,
    "theta0"      :   0,
    "psi0"        :   np.linspace(0.1, 0.9, N), #times psi_wall
    "z0"          :   0,
    "Pz0"         :   -0.035,
    "tspan"       :   np.linspace(0, 10000, 100000) # t0, tf, steps
    
}

# fmt: on
