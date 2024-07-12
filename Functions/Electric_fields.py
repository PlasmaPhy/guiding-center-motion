import numpy as np
import Functions.Qfactors as Qfactors


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

        # Field components
        Er = self.a * r**2 + self.b
        Etheta = 0
        Ezeta = 0

        # Derivatives of Φ with respect to ψ_π, θ
        psi = r**2 / 2
        Phi_der_psip = -self.q.q_of_psi(psi) * (self.a * r - self.b / r)
        Phi_der_theta = 0

        return np.array([Er, Etheta, Ezeta, Phi_der_psip, Phi_der_theta])

    def Er_of_r(self, r):  # Ready to commit
        """Returns the value of the field Er

        Args:
            r (float/array): the radial position(s)

        Returns:
            float/array: potential Φ value(s)
        """

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

    def extremums(self):  # Ready to commit
        """Returns

        Returns:
            _type_: _description_
        """
        Phimin = self.Phi_of_r(self.psi_wall)
        Phimax = self.Phi_of_r(0)
        return np.array([Phimin, Phimax])
