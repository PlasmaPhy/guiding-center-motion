import numpy as np
from dataclasses import dataclass
import orbit


@dataclass
class SignalAnalysis:
    t: np.ndarray
    theta: np.ndarray
    zeta: np.ndarray

    def run(self):
        """
        Check if signals are bounded or not
            * ζ is always unbounded, θ could be both.
            check if θ is periodic by checking if min and max are within 2π
            if θ is bounded:
                use event locator to find ωθ
                use theory to immediately find ωζ(=ωθ) and ωζ0 (from 2 peaks)
            if θ is unbounded:

        """

        trapped_theta = self.istrapped(self.theta)
        trapped_zeta = self.istrapped(self.zeta)

        if trapped_theta:
            self.event_frequency()

    def istrapped(signal):
        if np.ptp(signal) < 2 * np.pi:  # possible floating point error
            return True
        return False

    def event_frequency():
        solution = orbit()
