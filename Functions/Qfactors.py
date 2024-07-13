import numpy as np
from scipy.special import hyp2f1


class Unity:  # Ready to commit
    """Initializes an object q with "q(ψ) = 1" """

    which_ode = "both"

    def q_of_psi(self, psi):
        return 1

    def q_of_psip(self, psip):
        return 1

    def psi_from_psip(self, psip):
        return psip

    def psip_from_psi(self, psi):
        return psi


class Parabolic:  # Ready to commit
    """Ιnitializes an object q with "q(ψ) = 1 + ψ^2" """

    which_ode = "both"

    def q_of_psi(self, psi):
        return 1 + psi**2

    def q_of_psip(self, psip):
        psi = self.psi_from_psip(psip)
        return 1 + np.atan(psi) ** 2

    def psi_from_psip(self, psip):
        return np.tan(psip)

    def psip_from_psi(self, psi):
        return np.atan(psi)


class Hypergeometric:  # Ready to commit
    """Initializes an object q with "q = hypergeometric".

    Needs psi_wall as input upon initialization.
    """

    def __init__(self, psi_wall, q0=1.1, psi_knee=2.5, q_wall=3.5, n=2):
        self.psi_wall = psi_wall
        self.q0 = q0
        self.psi_knee = psi_knee
        self.q_wall = q_wall
        self.n = n

    def q_of_psi(self, psi):
        return self.q0 * (1 + (psi / (self.psi_knee / 1)) ** self.n) ** (1 / self.n)

    def q_of_psip(self, psip):
        psi = self.psi_from_psip(psip)
        return self.q_of_psi(psi)

    def psi_from_psip(self, psip):
        return None

    def psip_from_psi(self, psi):
        a = b = 1 / self.n
        c = 1 + 1 / self.n
        z = (1 - (self.q_wall / self.q0) ** self.n) * (psi / self.psi_wall) ** self.n
        return psi / self.q0 * hyp2f1(a, b, c, z)
