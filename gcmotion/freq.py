import numpy as np
from scipy.fftpack import rfft, rfftfreq
from scipy.signal import find_peaks as fp
import matplotlib.pyplot as plt


class FreqAnalysis:
    r"""Component of class ``Particle``. Contains all the fft calculation methods."""

    def __init__(self, cwp, x, angle, trim=True, normal=False, prominence=0.8):
        r"""Copies attributes from cwp to self.

        The instance itself is automatically initialized internally by the Particle
        class, and should not be called by the user.

        Args:
            cwp (Particle): The Current Working Particle
        """
        self.__dict__ = dict(cwp.__dict__)
        self.angle = angle
        self.x = x
        self.t = self.tspan.copy()
        self.trim = trim
        self.normal = normal
        self.prominence = prominence
        if not self.normal:
            self.t /= self.w0

    def run(self, prominence: float | int = 0.8):

        self.omega_manual, self.omega_manual_err = self._get_omega_manual()
        self.bias = self._get_bias(self.x)
        self._fft()
        print(self.harmonics[:10])

    def __str__(self):
        return f"{(self.omega_manual, self.omega_manual_err, self.bias)}"

    def _get_omega_manual(self) -> tuple[float, float]:

        # Find the major peaks,valleys using prominence
        peaks, _ = fp(self.x, prominence=self.prominence)  # Tune prominence to filter local peaks
        valleys, _ = fp(-self.x, prominence=self.prominence)

        # Get the time values of the detected peaks and valleys
        peak_times = self.t[peaks[1:-2]]
        valley_times = self.t[valleys[1:-2]]
        # print(f"Detected peaks at times: {peak_times[:10].astype(int)}")

        if self.trim:
            self._trim(peaks)

        # Calculate the periods by taking the time difference between consecutive peaks/valleys
        peak_periods = np.diff(peak_times)
        valley_periods = np.diff(valley_times)
        periods = np.concatenate((peak_periods, valley_periods))

        period = np.mean(periods)
        omega = 2 * np.pi / period
        omega_err = 2 * np.pi * np.std(periods) / (period**2)

        return (omega, omega_err)

    def _trim(self, peaks: np.ndarray):
        self.x = self.x[peaks[0] : peaks[-1]]
        self.t = self.t[peaks[0] : peaks[-1]]

    def _get_bias(self, x):
        grad = np.gradient(x)
        bias = np.mean(grad)
        return bias

    def _fft(self):

        # Unbiased x
        x = self.x - self.bias * self.t

        t0, tf, steps = self.t[0], self.t[-1], self.t.shape[0]
        dt = (tf - t0) / steps
        sr = 1 / dt  # Sample rate

        self.X = rfft(x)
        freqs = rfftfreq(len(x)) * sr
        self.omegas = 2 * np.pi * freqs

        self.xargmax = np.argmax(self.X)
        # first = self.omegas[self.xargmax]

        def harmonics():
            # Find the major peaks,valleys using prominence
            dist = self.xargmax
            peaks, _ = fp(
                self.X[self.xargmax :], distance=dist, prominence=0.5
            )  # Tune prominence to filter local peaks

            # Get the time values of the detected peaks and valleys
            peaks = self.omegas[peaks]

            return peaks

        self.harmonics = harmonics()
