import gcmotion as gcm
import numpy as np

R, a = 6.2, 2  # Major/Minor Radius in [m]
q = gcm.qfactor.Hypergeometric(R, a)
Bfield = gcm.bfield.LAR(i=0, g=1, B0=5)
Efield = gcm.efield.Radial(R, a, q, Ea=75000, minimum=0.9, waist_width=50)

species = "p"
mu = 10e-5
theta0 = np.pi / 3
psi0 = 0.5  # times psi_wall
z0 = np.pi
Pz0 = -0.025
t_eval = np.linspace(0, 100000, 10000)  # t0, tf, steps

init_cond = [theta0, psi0, z0, Pz0]
particle1 = gcm.Particle(species, mu, init_cond, t_eval, R, a, q, Bfield, Efield)
cwp = particle1

cwp.run()

# gcm.time_evolution(cwp, percentage=100)
# gcm.tokamak_profile(cwp)
# gcm.drift(cwp)
