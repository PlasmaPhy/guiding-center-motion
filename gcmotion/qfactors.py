"""
To add a new q-factor, simply copy-paste an already existing class
(idealy the Unity one) and fill the __init__() method with the 
parameters, and the q_of_psi() and psip_of_psi() to fit your q-factor.

.. note::
    Keep in mind that when those methods return singular values (rather than
    np.ndarrays), they should return a float, and not a np.float. This is mainly
    for optimization reason and should probably not cause problems.

The general structure is this::

    class MyQFactor:
        
        def __init__(self, **<parameters>):
            <set parameters>

        def q_of_psi(self, psi):
            > Calculates q(ψ). Return type should be same as input.
            > 
            > Used inside dSdt, Φ derivatives (returns a float) and plotting of
            > q factor (returns an np.ndarray).
            > 
            > Args:
            >     psi (float/list/array): Value(s) of ψ.
            > 
            > Returns:
            >     float/list/array: Calculated q(ψ)

            return <expression>

        def psip_of_psi(self, psi):
            Calculates ψ_p(ψ).

            > Used in calculating psip_wall in many methods (returns a float), in calculating
            > ψ_p's time evolution (returns an np.ndarray), in Energy contour calculation
            > (returns an np.ndarray) and in q-factor plotting (returns an np.ndarray)
            > 
            > Args:
            >     psi (float/np.ndarray): Value(s) of ψ.
            > 
            > Returns:
            >     float/np.ndarray: Calculated ψ_p(ψ).

            return <expression>
            
"""

import numpy as np
from scipy.special import hyp2f1


class Unity:  # Ready to commit
    """Initializes an object q with "q(ψ) = 1" """

    def q_of_psi(self, psi):
        return 1

    def psip_of_psi(self, psi):
        return psi


class Parabolic:  # Ready to commit
    """Initializes an object q with "q(ψ) = 1 + ψ^2" """

    def q_of_psi(self, psi):
        return 1 + psi**2

    def psip_of_psi(self, psi):
        return np.atan(psi)


class Hypergeometric:  # Ready to commit
    """Initializes an object q with "q = hypergeometric"."""

    def __init__(self, R, a, q0=1.1, psi_knee=2.5, n=2):
        self.r_wall = a / R
        self.psi_wall = (self.r_wall) ** 2 / 2  # normalized to R

        self.psi_wall = self.psi_wall
        self.q0 = q0
        self.psi_knee = 0.75 * self.psi_wall
        self.n = n
        self.q_wall = self.q_of_psi(self.psi_wall)

    def q_of_psi(self, psi):
        return self.q0 * (1 + (psi / (self.psi_knee)) ** self.n) ** (1 / self.n)

    def psip_of_psi(self, psi):
        a = b = 1 / self.n
        c = 1 + 1 / self.n
        z = (1 - (self.q_wall / self.q0) ** self.n) * (psi / self.psi_wall) ** self.n
        if type(psi) is float:
            return psi / self.q0 * float(hyp2f1(a, b, c, z))
        else:
            return psi / self.q0 * hyp2f1(a, b, c, z)
