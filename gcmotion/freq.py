import numpy as np
from math import tan
from scipy.fftpack import rfft, rfftfreq
from scipy.signal import find_peaks as fp
import matplotlib.pyplot as plt


class FreqAnalysis:
    r"""Component of class ``Particle``. Contains all the fft calculation methods."""

    def __init__(
        self,
        cwp,
        x: np.ndarray,
        angle: str,
        trim: bool = True,
        normal: bool = False,
        remove_bias: bool = False,
    ):
        r"""Copies attributes from cwp to self.

        All parameters are stored in self, to avoid confusion.

        The instance itself is automatically initialized internally by the Particle
        class, and should not be called by the user.

        Args:
            cwp (Particle): The Current Working Particle.
            x (np.ndarray): The timeseries to be analysed.
            trim (bool, optional): Whether or not to trim the edges of the time series.
                Defaults to True.
            normal (bool, optional): Whether or not to use normal or lab units.
                Defaults to True.
            remove_bias (bool, optional): Whether or not to remove the signal's bias.
                Defaults to True.
        """
        # Grab cwp's attributes
        self.__dict__ = dict(cwp.__dict__)

        self.angle = angle
        # self.x = np.concatenate([x, x[1:][::-1]])
        # self.t = np.concatenate([self.tspan.copy(), self.tspan[-1] + self.tspan.copy()[1:]])
        self.x = x
        self.t = self.tspan.copy()
        self.trim = trim
        self.normal = normal
        self.remove_bias = remove_bias
        # Normalise tspan once and for all
        if not self.normal:
            self.t /= self.w0
            self.freq_unit = "Hz"
            self.time_unit = "s"
        else:
            self.freq_unit = "[normalised frequency units]"
            self.time_unit = "[normalised time unit]"

        self.zeroth_freq = None
        # Flag to know if they frequencies are calculated correctly
        self.q_kinetic_ready = self.trim and self.remove_bias

        self.run()

    def run(self):
        r"""Run the analysis in the correct order."""

        if not self._singal_ok():
            print(
                "Error.  The signal might be aperiodic, or it didnt have time to"
                + "complete enough cycles."
            )
            self.signal_ok = False
            return
        else:
            self.signal_ok = True

        self._biases()

        if self.remove_bias:
            self.x -= self.dc_bias + self.zeroth_freq * self.t

        self.omega_manual, self.omega_manual_err = self._get_omega_manual()

        if self.trim:
            self._trim()

        self._fft()

    def _singal_ok(self):
        """Checks if the singal is actually periodic or not.

        It requires the signal to have at least 4 local maximums.
        Else, the signal might be aperiodic, or it didnt have time to
        complete enough cycles.
        """

        peaks, _ = fp(self.x)
        if len(peaks) < 4:
            return False
        else:
            return True

    def __str__(self):
        r"""Calculate the results strings.

        Returns:
            str: The results string to be printed.
        """

        info_str = (
            f"---------{self.angle.upper()}---------\n"
            + f"Frequency (from peak-to-peak)\t\t= ({self.omega_manual:.4g} \u00b1 {self.omega_manual_err:.4g}) {self.freq_unit}.\n"
            + f"DC bias:\t\t\t\t= {self.dc_bias:.4g} rads.\n"
            + f"Bias (zeroth frequency)\t\t\t= {self.zeroth_freq:.4g} rads/{self.time_unit}.\n"
            + f"Filtered(unbiased) signal amplitude\t= {self.amplitude:.4e} rads.\n"
        )
        return info_str

    def _get_omega_manual(self) -> tuple[float, float]:
        r"""Calculate the frequencies by measuring the distances between peaks and between valleys
        and taking their mean value.

        Returns:
            tuple[float, float]: the calculated frequency and its standard error.
        """

        height = self.amplitude * 0.8

        # Find the major peaks,valleys
        self.peaks, _ = fp(self.x, height=height)
        valleys, _ = fp(-self.x, height=height)

        # Get the time values of the detected peaks and valleys
        peak_times = self.t[self.peaks[1:-2]]
        valley_times = self.t[valleys[1:-2]]

        # Calculate the periods by taking the time difference between consecutive peaks/valleys
        peak_periods = np.diff(peak_times)
        valley_periods = np.diff(valley_times)
        periods = np.concatenate((peak_periods, valley_periods))

        period = np.mean(periods)
        omega = 2 * np.pi / period
        omega_err = 2 * np.pi * np.std(periods) / (period**2)

        return (omega, omega_err)

    def _trim(self):
        r"""Trims the edges of the time series, so that it includes an integer amount
        of full cycles. That way the FFT is much more clean and accurate.
        """

        self.x = self.x[self.peaks[0] : self.peaks[-1]]
        self.t = self.t[self.peaks[0] : self.peaks[-1]]

    def _biases(self):
        """Calculates the DC and rising (zeroth frequency) biases.

        Does not subtract them from the signal. This happens in ``__init__()`` only if the
        parameter ``remove_bias`` is True.
        """
        # Rising bias
        t = self.t.copy()
        x = self.x.copy()

        # Calculate zeroth frequency usning 2 consecutive peaks:
        peaks, _ = fp(x)
        peak_indices = peaks[0:2]
        peak_times = t[peak_indices]
        peak_signal = x[peak_indices]
        hdist = np.diff(peak_times)
        vdist = np.diff(peak_signal)
        self.zeroth_freq = float(vdist / hdist)

        # Trim a copy of self.x to calculate the DC bias accurately, without changing self.x yet.
        x -= self.zeroth_freq * t  # Remove rising bias

        peaks, _ = fp(x)
        x = x[peaks[0] : peaks[-1]]
        self.dc_bias = np.mean(x)
        x -= self.dc_bias

        # Now calculate amplitude correctly
        self.amplitude = np.abs(x).max()

    def _fft(self):
        r"""Calculates the FFT of the timeseries.

        .. note:: Even though the time evolution plot only shows a few periods
            for clarity, the FFT is calculated across the full orbit.

        """

        t0, tf, steps = self.t[0], self.t[-1], self.t.shape[0]
        dt = (tf - t0) / steps
        sr = 1 / dt  # Sample rate

        self.X = rfft(self.x)
        freqs = rfftfreq(len(self.x)) * sr
        self.omegas = 2 * np.pi * freqs

        self.xargmax = np.argmax(self.X)
        # first = self.omegas[self.xargmax]

        def harmonics():
            """Calculates the harmonics of the signal, starting from the
            fundamental.

            This only runs if the bias is removed.

            .. todo:: Find a way to make it work even without removing
                the bias.

            Returns:
                np.ndarray: the harmonics' frequencies
            """
            # Find the major peaks,valleys using prominence
            height = self.X[self.xargmax]
            peaks, _ = fp(self.X, height=height / 100)

            peaks = self.omegas[peaks]

            return peaks

        if self.remove_bias:
            self.harmonics = harmonics()

    def get_omegas(self):
        r"""Returns the 0th and 1st frequencies, needed to calculate the q_kinetic.

        .. note :: Only runs if the frequencies are calculated correctly (trim = True
            and remove_bias = True).

        Returns:
            tuple: The calculated frequencies (or `None` if not calculated correctly.)
        """
        if self.q_kinetic_ready and self.signal_ok:
            return self.zeroth_freq, self.harmonics[1]
        else:
            return (None, None)
