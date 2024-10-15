import numpy as np
from time import time
from scipy.fftpack import rfft, rfftfreq
from scipy.signal import find_peaks as fp
from numpy.polynomial import Polynomial


class FreqAnalysis:

    def __init__(
        self,
        cwp,
        angle: str,
        sine: bool = False,
        trim_params: dict = {},
        remove_bias: bool = True,
    ):
        # Grab cwp's attributes
        self.__dict__ = dict(cwp.__dict__)
        self._orbit = cwp._orbit
        self.event = [cwp.events("single_theta_period")]

        # Parameters
        self.angle = angle
        self.sine = sine
        self.trim_params = trim_params
        self.remove_bias = remove_bias

    def run(self):
        """Runs the methods in the correct order"""
        self._find_frequencies()

        # FFT data
        if self.trim_params:
            self.signal, self.t_signal, self.t_events = self._trim_signal()
        else:
            self.signal = getattr(self, self.angle)
            self.t_signal = self.t_eval.copy()
            self.trim_str = ""

        # Lab time units
        self.t_signal = np.array(self.t_signal / self.w0)

        # If zeta given we use ζ - ζ_hat = ζ - ω0ζ*t for FFT
        if self.angle == "zeta":
            p = Polynomial.fit(self.t_signal, self.signal, 1)
            self.a = p.convert().coef[1]  # Just for sanity check, not used
            self.b = p.convert().coef[0]  # DC Bias, will subtract for beter FFT
            self.signal = self.signal - self.zeta_0freq_event * self.t_signal - self.b

        self._fft()
        self._fft_peaks()

    def __repr__(self):

        polynomial = ""
        if self.angle == "zeta":
            # Polynomial Results
            polynomial = (
                "-------Linear fitting to find ωζ0 and DC bias--------\n\n"
                + f"Linear coefficient a   = zeroth ζ frequency\t= {self.a:.3e}\n"
                + f"Constant coefficient b = DC Bias \t\t= {self.b:.3e}\n\n"
            )

        # Event locator results
        single_period = (
            "----------------Running a single period (event locator)----------------\n\n"
            + f"\tCalculated θ frequency\t\t\t= {self.theta_freq_event:.4g}\n"
            + f"\tCalculated zeroth ζ frequency\t\t= {self.zeta_0freq_event:.4g}\n"
            + f"\tCalculated ζ fast frequency\t\t= {self.zeta_freq_event:.4g}\n"
            + f"\nCalculation time: {self.single_period_orbit_duration}s.\n\n"
        )

        # FFT results
        fft_results = "-----------------------------FFT results-------------------------------\n\n"
        if self.angle == "zeta":
            fft_results += "Actually using as signal for the FFT ζ(t) - ζ_hat(t) = ζ(t) - t*ωζ0_hat - DC Bias\nand not ζ(t)\n\n"

        if self.trim_params:
            periods = self.trim_params["periods"]
            steps_per_period = self.trim_params["steps_per_period"]

            fft_results += (
                "Using trimmed signal.\n"
                + f"\tTotaling {periods} periods, {steps_per_period} samples each, for a total\n"
                + f"\tof {steps_per_period*periods} samples, and a sample rate of {self.sr:.4g} samples/s.\n\n"
            )
        else:
            fft_results += "Using the whole signal.\n\n"

        if self.angle == "theta":
            harmonics = [f"{x:.4g}Hz" for x in self.theta_harmonics_fft[:5]]
            fft_results += (
                f"\tCalculated θ frequency\t\t\t= {self.theta_freq_fft:.4g}\n"
                + f"\t{len(self.fft_peak_index)} harmonics found: {harmonics}\n"
            )
        elif self.angle == "zeta":
            harmonics = [f"{x:.4g}Hz" for x in self.zeta_harmonics_fft[:5]]
            fft_results += (
                f"\tCalculated ζ(t) - ζ_hat(t) zeroth frequency\t\t= {self.zeta_0freq_fft:.4g}\n"
                # + f"\tCalculated z fast frequency\t\t\t= {self.zeta_freq_fft:.4g}\n"
                + f"\t{len(harmonics)} harmonics found: {harmonics}\n"
            )

        fft_results += (
            f"\n\tSample rate = {self.sr:.4g} samples/s."
            + f"\nCalculation time: {self.fft_duration}s.\n\n"
        )

        return polynomial + single_period + fft_results

    def _find_frequencies(self):
        """Finds the signal base frequencies by running a single period orbit
        using an event locator.
        """

        start = time()
        _, _, _, z, _, _, _, t_events, t_eval, dz = self._orbit(events=self.event)
        end = time()
        self.single_period_orbit_duration = f"{end-start:.4f}"

        t_events = np.array(t_events).flatten()

        # For some psi0s, Pz0s theta is quasi periodic as well--> events never satisfied
        # --> t_events = [], only used for ωθ(Ρζ) do not pay attention
        if len(t_events) != 2:
            print("t_events = []. Change initial conditions")
            return

        # Theta period
        self.theta_period_event_NU = float(np.diff(t_events))
        self.theta_freq_event_NU = self.zeta_freq_event_NU = 2 * np.pi / self.theta_period_event_NU

        # Zeta period
        # dz = abs(np.diff(np.interp(t_events, t_eval, z))[0])
        print(f"dz: {dz}")
        print(f"Wrong/Old dz: {abs(np.diff(np.interp(t_events, t_eval, z))[0])}")
        zeta_period_event_NU = 2 * np.pi * self.theta_period_event_NU / dz
        self.zeta_0freq_event_NU = 2 * np.pi / zeta_period_event_NU

        # To lab time unites
        self.theta_period_event = self.theta_period_event_NU / self.w0
        self.zeta_period_event = zeta_period_event_NU / self.w0
        self.theta_freq_event = self.zeta_freq_event = 2 * np.pi / self.theta_period_event
        self.zeta_0freq_event = self.zeta_0freq_event_NU * self.w0
        print(self.zeta_0freq_event)

    def _trim_signal(self):
        """Recalculates an orbit with pre-defined number of periods and sample rate.

        _find_frequencies must run first.

        Returns:
            tuple: tuple containing the calculated signal.
        """

        self.periods = self.trim_params.get("periods", 1)
        self.steps_per_period = self.trim_params.get("steps_per_period", 1)

        # Setup time span
        t0, tf, steps = (
            self.t_eval[0],
            self.theta_period_event_NU * self.periods,
            self.steps_per_period * self.periods,
        )
        t_eval = np.linspace(t0, tf, steps)

        # Setup stop event
        def stop_event(t, S):
            return (S[0] - self.theta0) or (S[1] - self.psi0)

        stop_event.terminal = self.periods + 1
        stop_event.direction = 1

        # Calculate the orbit
        start = time()
        theta, _, _, zeta, _, _, _, t_events, t_signal, _ = self._orbit(
            events=stop_event, t_eval=t_eval
        )
        end = time()
        self.dft_orbit_duration = f"{end-start:.4f}"
        t_events = np.array(t_events).flatten()
        signal = eval(self.angle)

        return signal, t_signal, t_events

    def _fft(self):
        """Calculates the DFT of the sine of the signal"""

        t0, tf, steps = self.t_signal[0], self.t_signal[-1], self.t_signal.shape[0]
        dt = (tf - t0) / steps
        self.sr = 1 / dt  # Sample rate

        # signal = np.exp(1j * self.signal)
        # signal = np.sin(self.signal)

        start = time()

        self.X = np.abs(rfft(self.signal))
        self.omegas = 2 * np.pi * rfftfreq(len(self.signal), d=dt)

        end = time()
        self.fft_duration = f"{end-start:.4f}"
        return

    def _fft_peaks(self):
        """Does its best to find the peaks of the DFT."""

        heighest_peak = self.X.max()
        height = heighest_peak * 0.02
        self.fft_peak_index, properties = fp(np.abs(self.X), height=height, distance=50)

        if self.angle == "theta":
            self.theta_freq_fft = self.base_freq = self.omegas[self.fft_peak_index[0]]
            self.theta_harmonics_fft = self.omegas[self.fft_peak_index]
            self.found_indeces = self.fft_peak_index

        elif self.angle == "zeta":
            self.zeta_0freq_fft = self.omegas[self.fft_peak_index[0]]

            self.zeta_freq_fft = self.base_freq = (
                self.omegas[self.fft_peak_index[1]] + self.zeta_0freq_fft
            )

            # Source(s): dude trust me
            self.zeta_harmonics_fft = [
                self.omegas[int(i)] + self.zeta_0freq_fft for i in self.fft_peak_index[1::2]
            ]

            freq_pairs = self.fft_peak_index[1:]
            fast_indeces = [
                int((freq_pairs[i + 1] + freq_pairs[i]) / 2)
                for i in range(0, len(freq_pairs) - 1, 2)
            ]
            self.found_indeces = fast_indeces
