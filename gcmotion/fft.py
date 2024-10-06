import numpy as np
from scipy.fftpack import rfft, rfftfreq
from scipy.signal import argrelmax


class FreqAnalysis:
    r"""Component of class ``Particle``. Contains all the fft calculation methods."""

    def __init__(self, cwp):
        r"""Copies attributes from cwp to self.

        The instance itself is automatically initialized internally by the Particle
        class, and should not be called by the user.

        Args:
            cwp (Particle): The Current Working Particle
        """
        self.__dict__ = dict(cwp.__dict__)

    def from_peaks(self, x):
        extrema = argrelmax(x, mode="wrap")[0]  # Indeces
        if len(extrema) <= 1:
            print("Not enough periods.")
            return
        elif len(extrema) >= 4:
            extrema = extrema[1:-1]  # Trim first and last vaules
        else:
            print("Too few periods, but continuing...")

        tmax = self.tspan[extrema]
        tdiffs = np.diff(tmax)
        period_avg = np.mean(tdiffs)
        period_err = np.std(tdiffs)

        f_avg = 1 / period_avg
        f_err = period_err / period_avg**2  # Error propagation
        counts = len(tdiffs)
        return (f_avg, f_err, counts)

    def _fft(self, x):
        t0, tf, steps = self.tspan[0], self.tspan[-1], self.tspan.shape[0]
        dt = (tf - t0) / steps
        sr = 1 / dt  # Sample rate

        X = rfft(x)
        freqs = rfftfreq(len(x)) * sr

        return (X, freqs)
