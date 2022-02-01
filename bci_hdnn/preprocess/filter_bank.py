import numpy as np
import scipy.signal as signal
from scipy.signal import cheb2ord


class FilterBank:
    def __init__(self, fs, nb_bands=9, f_width=4.0, f_min=4.0, f_max=40.0):
        """Filter Bank class. __call__ method will apply 
        passband Chebyshev type II filters.

        Parameters
        ----------
        fs : float
            sample frequency in Hz
        f_width : float
            Passband bandwidth in Hz, by default 4
        f_min : int, optional
            Min frequency in Hz, by default 4
        f_max : int, optional
            Max frequency in Hz, by default 40
        """
        self.B = nb_bands
        self.fs = fs
        self.f_min = f_min
        self.f_max = f_max
        self.f_width = f_width
        self.f_trans = 2  # transition width in Hz
        self.f_pass = np.linspace(f_min, f_max, nb_bands, endpoint=False)
        # self.f_pass = np.arange(f_min, f_max, f_width)
        self.gpass = 3
        self.gstop = 40
        self.filter_coeff, self.filter_freqs = self.get_filter_coeff()

    def get_filter_coeff(self):
        """Initiate filter coefficients for Filter Bank

        Returns
        -------
        Tuple[dict, dict]
            - filter_coeff: list of dict
                "a", "b": np.ndarray containing filter coefficients
            - filter_freqs: list of dict
                "pass": passpand frequencies in Hz
                "stop": stopband frequencies in Hz
        """
        filter_coeff = []
        filter_freqs = []
        Nyquist_freq = self.fs/2
        for i, f_low_pass in enumerate(self.f_pass):
            f_pass = np.asarray([f_low_pass, f_low_pass+self.f_width])
            f_stop = np.asarray(
                [f_pass[0]-self.f_trans, f_pass[1]+self.f_trans])
            wp = f_pass/Nyquist_freq
            ws = f_stop/Nyquist_freq
            order, wn = cheb2ord(wp, ws, self.gpass, self.gstop)
            b, a = signal.cheby2(order, self.gstop, wn, btype='bandpass')
            filter_coeff.append({'b': b, 'a': a})
            filter_freqs.append({"pass": f_pass, "stop": f_stop})
        return filter_coeff, filter_freqs

    def filter_data(self, eeg_data: np.ndarray) -> np.ndarray:
        """Filter data by Filter Bank

        Parameters
        ----------
        eeg_data : np.ndarray
            EEG signals, shape (N, C, T)
            N trials, C channels, T discrete time

        Returns
        -------
        np.ndarray
            Filtered signals, shape (B, N, C, T)
            B number of filter bands
        """
        n_trials, n_channels, n_samples = eeg_data.shape
        filtered_data = np.zeros(
            (len(self.filter_coeff), n_trials, n_channels, n_samples))
        for i, filter_ab in enumerate(self.filter_coeff):
            b = filter_ab.get('b')
            a = filter_ab.get('a')
            filtered_data[i] = signal.lfilter(b, a, eeg_data)
        return filtered_data

    def __call__(self, eeg_data):
        """Wrapper for filter_data method
        """
        return self.filter_data(eeg_data)
