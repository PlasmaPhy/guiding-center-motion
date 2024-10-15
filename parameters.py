import gcmotion as gcm
import numpy as np

# To be passed to objects as parameters for ease
R = 6.2
a = 2
q = gcm.qfactor.Hypergeometric(R, a)
N = 25  # Number of particles

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
    "psi0"        :   0.8,#np.linspace(0.3, 0.9, N), #times psi_wall
    "z0"          :   0,
    "Pz0"         :   np.linspace(-0.04721, -0.01892, N), #-0.025, (-0.03,-0.01, N)
    "t_eval"      :   np.linspace(0, 10000000, 150000) # t0, tf, steps

    
}

# fmt: on
