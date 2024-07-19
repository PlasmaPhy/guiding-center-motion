import numpy as np
from scipy.special import erf


class Nofield:
    """Initializes an electric field of 0

    Exists to avoid compatibility issues.
    """

    def Phi_der_of_psi(self, psi):
        return np.array([0, 0])

    def Er_of_psi(self, psi):
        return 0 * psi

    def Phi_of_r(self, r):
        return 0 * r

    def Phi_of_psi(self, psi):
        return 0 * psi

    def extremums(self):
        return np.array([0, 0])


class Parabolic:
    """Initializes an electric field of the form E = ar^2 + b"""

    def __init__(self, psi_wall, q):
        self.a = -1
        self.b = 0
        self.q = q
        self.psi_wall = psi_wall
        self.psip_wall = q.psip_from_psi(psi_wall)

    def Phi_der_of_psi(self, psi):
        """Calculates the E field values and certain derivatives of Φ.

        This function should only be used inside dSdt.

        Args:
            r (float): radial position

        Returns:
            np.array: calculated componets
        """

        # Derivatives of Φ with respect to ψ_π, θ
        r = np.sqrt(2 * psi)
        Phi_der_psip = -self.q.q_of_psi(psi) * (self.a * r - self.b / r)
        Phi_der_theta = 0

        return np.array([Phi_der_psip, Phi_der_theta])

    def Er_of_psi(self, psi):
        """Returns the value of the field Er

        Args:
            r (float/array): the radial position(s)

        Returns:
            float/array: potential Φ value(s)
        """
        r = np.sqrt(2 * psi)
        Er = self.a * r**2 + self.b
        return Er

    def Phi_of_r(self, r):
        """Returns the value of the potential Φ

        Args:
            r (float/array): the radial position(s)

        Returns:
            float/array: potential Φ value(s)
        """

        Phi = -(self.a / 3) * r**3 - self.b * r
        return Phi

    def Phi_of_psi(self, psi):
        """Returns the value of the potential Φ

        Args:
            psi (float/array): the mangetic surface label

        Returns:
            float/array: potential Φ value(s)
        """
        r = np.sqrt(2 * psi)
        Phi = -(self.a / 3) * r**3 - self.b * r
        return Phi

    def extremums(self):
        """Returns

        Returns:
            _type_: _description_
        """
        # Doesn't work yet
        Phimin = self.Phi_of_r(self.psi_wall)
        Phimax = self.Phi_of_r(0)
        return np.array([Phimin, Phimax])


class Radial:
    """Initializes an electric field of the form:
    E(r) = -Ea*exp(-(r-r_a)^2 / r_w^2))

    Caution if you change Ea's field. You must adjust extremums()
    accordingly.
    """

    def __init__(self, psi_wall, q, Ea=75):
        self.Ea = Ea  # kV/m
        self.r0 = np.sqrt(2 * psi_wall)
        self.ra = 0.9 * self.r0
        self.Efield_min = self.ra**2 / 2
        self.rw = self.r0 / 50  # waist, not wall
        self.psia = self.ra**2 / 2
        self.psiw = self.rw**2 / 2  # waist, not wall

        # Square roots, makes it a bit faster
        self.sr_psia = np.sqrt(self.psia)
        self.sr_psiw = np.sqrt(self.psiw)

        self.q = q
        self.psi_wall = psi_wall
        self.psip_wall = q.psip_from_psi(psi_wall)

    def Phi_der_of_psi(self, psi):

        # Derivatives of Φ(ψ) with respect to ψ_π, θ
        Phi_der_psip = (
            self.q.q_of_psi(psi)
            * self.Ea
            / (np.sqrt(2 * psi))
            * np.exp(-((np.sqrt(psi) - self.sr_psia) ** 2) / self.psiw)
        )
        Phi_der_theta = 0

        return np.array([Phi_der_psip, Phi_der_theta])

    def Er_of_psi(self, psi):
        r = np.sqrt(2 * psi)
        Er = -self.Ea * np.exp(-((r - self.ra) ** 2) / self.rw**2)
        return Er

    def Phi_of_r(self, r):
        psi = r**2 / 2
        Phi = self.Phi_of_psi(psi)
        return Phi

    def Phi_of_psi(self, psi):
        Phi = (
            self.Ea
            * np.sqrt(np.pi * self.psiw / 2)
            * (erf((np.sqrt(psi) - self.sr_psia) / self.sr_psiw) + erf(self.sr_psia / self.sr_psiw))
        )
        return Phi

    def extremums(self):
        Phimin = self.Phi_of_psi(0)
        Phimax = self.Phi_of_psi(self.psi_wall)
        return np.array([Phimin, Phimax])
