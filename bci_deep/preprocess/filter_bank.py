from typing import Tuple
import numpy as np
import scipy.signal as signal
from scipy.signal import cheb2ord
import torch
import torch.nn as nn


class FilterBank:
    def __init__(self, fs, nb_bands=9, 
                 f_width=4.0, f_min=4.0, f_max=40.0, f_trans=2.0,
                 gpass=3.0, gstop=30.0, **kwargs):
        """Filter Bank to apply passband Chebyshev type II filters

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
        super().__init__()
        self.B = nb_bands
        self.fs = fs
        self.f_min = f_min
        self.f_max = f_max
        self.f_width = f_width
        self.f_trans = f_trans  # transition width in Hz
        self.f_pass = np.linspace(f_min, f_max, nb_bands, endpoint=False)
        # self.f_pass = np.arange(f_min, f_max, f_width)
        self.gpass = gpass
        self.gstop = gstop
        self.get_filter_coeff()

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
        self.a_coeffs = []
        self.b_coeffs = []
        self.filter_freqs = []
        Nyquist_freq = self.fs/2

        for i, f_low_pass in enumerate(self.f_pass):
            f_pass = np.asarray([f_low_pass, f_low_pass+self.f_width])
            f_stop = np.asarray(
                [f_pass[0]-self.f_trans, f_pass[1]+self.f_trans])
            wp = f_pass/Nyquist_freq
            ws = f_stop/Nyquist_freq
            order, wn = cheb2ord(wp, ws, self.gpass, self.gstop)
            b, a = signal.cheby2(order, self.gstop, wn, btype='bandpass')
            self.a_coeffs.append(a)
            self.b_coeffs.append(b)
            self.filter_freqs.append({"pass": f_pass, "stop": f_stop})

        self.a_coeffs = np.stack(self.a_coeffs)
        self.b_coeffs = np.stack(self.b_coeffs)
        self.a_coeffs = torch.tensor(self.a_coeffs, requires_grad=False)
        self.b_coeffs = torch.tensor(self.b_coeffs, requires_grad=False)


    def np_forward(self, x: np.ndarray) -> np.ndarray:
        """Filter data by Filter Bank, use numpy function

        Parameters
        ----------
        eeg_data : np.ndarray
            EEG signals, shape (..., T)
            T discrete time

        Returns
        -------
        np.ndarray
            Filtered signals, shape (..., B, T)
            B number of filter bands
        """
        xfb = []
        a_coeffs = self.a_coeffs.numpy()
        b_coeffs = self.b_coeffs.numpy()
        xfb = [signal.lfilter(b, a, x) for a, b in zip(a_coeffs, b_coeffs)]
        return np.stack(xfb, -2)
