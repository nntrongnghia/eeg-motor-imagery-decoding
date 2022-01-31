import numpy as np

from .filter_bank import FilterBank
from .csp import CSP


class OVR_FBCSP:
    def __init__(self,
                 nb_classes: int, fs: float,
                 f_width=4.0, f_min=4.0, f_max=40.0, m_filters=2):
        """One Versus Rest - Filter Bank Common Spatial Pattern

        Parameters
        ----------
        nb_classes: int
            Number of classes
        fs : float
            Sample frequency in Hz
        f_width : float
            Passband bandwidth in Hz, by default 4
        f_min : int, optional
            Min frequency in Hz, by default 4
        f_max : int, optional
            Max frequency in Hz, by default 40
        m_filters : int, optional
            Number of CSP filters, by default 2
        """
        self.filter = FilterBank(fs, f_width, f_min, f_max)
        # CSP for each class of each filter band
        self.csps = [[CSP(m_filters) for _ in range(nb_classes)]
                     for _ in self.filter.filter_freqs]
        self.nb_classes = nb_classes
        self.m_filters = m_filters
        # number of passband filters
        self.B = len(self.filter.filter_freqs)
        # Transformation matrix (B, C, m*2*nb_classes)
        # will be initialized by self.fit
        self.W = None
        # Transposed W with expanded dims, ready to use in self.transform
        self.WT = None

    def get_csp_ovr(self, x_data: np.ndarray, y_labels: np.ndarray):
        pass

    def fit(self, x_data: np.ndarray, y_labels: np.ndarray):
        """Find the transformation matrix for OVR-FBCSP 
        This matrix will be stored in self.W, with shape (B, C, m*2*nb_classes)
        B number of passband filters, C channels, m number of CSP filters.

        Parameters
        ----------
        x_data : np.ndarray
            EEG signals, shape (N, C, T)
            N trials, C channels, T discrete time
        y_labels : np.ndarray
            EEG labels, shape (N, )
        """
        C = x_data.shape[-2]
        self.W = np.zeros((self.B, C, self.m_filters*2*self.nb_classes))
        classes = np.unique(y_labels)
        x_fb = self.filter.filter_data(x_data)  # (B, N, C, T)
        assert len(classes) == self.nb_classes, \
            f"y_labels should contain {self.nb_classes} classes"
        # TODO: optimize code to remove for loops ?
        for b in len(self.filter.filter_coeff):
            x = x_fb[b]  # (N, C, T)
            for cls in classes:
                y_ovr = (y_labels == cls).astype(np.int)
                self.csps[b][cls].fit(x, y_ovr)
            self.W[b] = np.concatenate([
                csp.Wb for csp in self.csps[b]
            ], axis=-1)
        self.WT = np.moveaxis(self.W, -2, -1)[None]

    def transform(self, x_data: np.ndarray) -> np.ndarray:
        """Feature extraction using OVR-FBCSP algorithm

        Parameters
        ----------
        x_data : np.ndarray
            EEG signals, shape (N, C, T)
            N trials, C channels, T discrete time

        Returns
        -------
        np.ndarray
            Feature matrix, shape (N, B, M)
            with M = 2*m*nb_classes
        """
        Z = self.WT @ x_data[:, None] # (N, B, M, T)
        ZT = np.moveaxis(Z, -2, -1)
        ZZt = Z @ ZT # (N, B, M, M)
        diag = np.diagonal(ZZt, axis1=-2, axis2=-1)
        return np.log(diag/diag.sum(-1, keepdims=True))

