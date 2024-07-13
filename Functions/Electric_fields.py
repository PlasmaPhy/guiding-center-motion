import numpy as np
import Functions.Qfactors as Qfactors
from scipy.special import erf


class Parabolic:
    """Initializes an electric field of the form E = ar^2 + b"""

    def __init__(self, psi_wall, q=Qfactors.Unity()):  # Ready to commit
        self.a = 1
        self.b = 0
        self.q = q
        self.psi_wall = psi_wall
        self.psip_wall = q.psip_from_psi(psi_wall)

    def orbit(self, r):  # Ready to commit
        """Calculates the E field values and certain derivatives of Φ.

        This function should only be used inside dSdt.

        Args:
            r (float): radial position

        Returns:
            np.array: calculated componets
        """

        # Derivatives of Φ with respect to ψ_π, θ
        psi = r**2 / 2
        Phi_der_psip = -self.q.q_of_psi(psi) * (self.a * r - self.b / r)
        Phi_der_theta = 0

        return np.array([Phi_der_psip, Phi_der_theta])

    def Er_of_psi(self, psi):  # Ready to commit
        """Returns the value of the field Er

        Args:
            r (float/array): the radial position(s)

        Returns:
            float/array: potential Φ value(s)
        """
        r = np.sqrt(2 * psi)
        Er = self.a * r**2 + self.b
        return Er

    def Phi_of_r(self, r):  # Ready to commit
        """Returns the value of the potential Φ

        Args:
            r (float/array): the radial position(s)

        Returns:
            float/array: potential Φ value(s)
        """

        Phi = -(self.a / 3) * r**3 - self.b * r
        return Phi

    def Phi_of_psi(self, psi):  # Ready to commit
        """Returns the value of the potential Φ

        Args:
            psi (float/array): the mangetic surface label

        Returns:
            float/array: potential Φ value(s)
        """
        r = np.sqrt(psi)
        Phi = -(self.a / 3) * r**3 - self.b * r
        return Phi

    def extremums(self):  # Ready to commit
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
    E(r) = -Ea*exp(-(r-r_a)^2 / r_w^2))"""

    def __init__(self, psi_wall, q=Qfactors.Unity()):  # Ready to commit
        self.Ea = 75  # kV/m
        self.r0 = 1
        self.ra = 0.98 * self.r0
        self.rw = self.r0 / 50  # waist, not wall
        self.psia = self.ra**2 / 2
        self.psiw = self.rw**2 / 2  # waist, not wall

        # Square roots, makes it a bit faster
        self.sr_psia = np.sqrt(self.psia)
        self.sr_psiw = np.sqrt(self.psiw)

        self.q = q
        self.psi_wall = psi_wall
        self.psip_wall = q.psip_from_psi(psi_wall)

    def orbit(self, r):  # Ready to commit

        # Derivatives of Φ with respect to ψ_π, θ
        psi = r**2 / 2
        Phi_der_psip = (
            self.q.q_of_psi(psi)
            * self.Ea
            / (np.sqrt(2) * psi)
            * np.exp(-((np.sqrt(psi) - self.sr_psia) ** 2) / self.psiw)
        )
        Phi_der_theta = 0

        return np.array([Phi_der_psip, Phi_der_theta])

    def Er_of_psi(self, psi):  # Ready to commit
        Er = -self.Ea * np.exp(-((np.sqrt(2 * psi) - self.ra) ** 2) / self.rw**2)
        return Er

    def Phi_of_r(self, r):  # Ready to commit
        Phi = (
            self.Ea
            * np.sqrt(np.pi * self.psiw / 2)
            * (
                erf((r / np.sqrt(2) - self.sr_psia) / self.sr_psiw)
                + erf(self.sr_psia / self.sr_psiw)
            )
        )
        return Phi

    def Phi_of_psi(self, psi):  # Ready to commit
        Phi = (
            self.Ea
            * np.sqrt(np.pi * self.psiw / 2)
            * (erf((np.sqrt(psi) - self.sr_psia) / self.sr_psiw) + erf(self.sr_psia / self.sr_psiw))
        )
        return Phi

    def extremums(self):  # Ready to commit
        # Doesn't work yet
        Phimin = self.Phi_of_r(self.psi_wall)
        Phimax = self.Phi_of_r(0)
        return np.array([Phimin, Phimax])
