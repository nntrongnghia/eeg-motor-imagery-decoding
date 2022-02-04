from math import sqrt
import numpy as np
import torch
import torch.nn as nn
from bci_hdnn.preprocess.filter_bank import FilterBank
from bci_hdnn.preprocess.csp import CSP


class OVR_CSP(nn.Module):
    def __init__(self, nb_classes: int, m_filters=2, trainable=True):
        """One Versus Rest - Common Spatial Pattern

        Parameters
        ----------
        nb_classes: int
            Number of classes
        m_filters : int, optional
            Number of CSP filters, by default 2
        """
        super().__init__()
        self.trainable = trainable
        # CSP for each class of each filter band
        self.csps = None
        self.nb_classes = nb_classes
        self.m_filters = m_filters
        # Transformation matrix (B, C, m*2*nb_classes)
        # will be initialized by self.fit
        self.W = None
        # Transposed W with expanded dims, ready to use in self.transform
        self.WT = None


    def fit(self, x_fb: np.ndarray, y_labels: np.ndarray):
        """Find the transformation matrix for OVR-FBCSP 
        This matrix will be stored in self.W, with shape (B, C, m*2*nb_classes)
        B number of passband filters, C channels, m number of CSP filters.

        Parameters
        ----------
        x_fb : np.ndarray
            EEG signals filtered by B passband, shape (B, N, C, T)
            N trials, C channels, T discrete time
        y_labels : np.ndarray
            EEG labels, shape (N, )
        """
        if isinstance(x_fb, torch.Tensor):
            x_fb = x_fb.detach().cpu().numpy()
        B, N, C, T = x_fb.shape
        self.csps = [[CSP(self.m_filters) for _ in range(self.nb_classes)] for _ in range(B)]
        self.W = np.zeros((B, C, self.m_filters*2*self.nb_classes))
        classes = np.unique(y_labels)
        assert len(classes) == self.nb_classes, \
            f"y_labels should contain {self.nb_classes} classes"
        # TODO: optimize code to remove for loops ?
        for b in range(B):
            x = x_fb[b]  # (N, C, T)
            for cls in classes:
                y_ovr = (y_labels == cls).astype(np.int)
                self.csps[b][cls].fit(x, y_ovr)
            self.W[b] = np.concatenate([
                csp.Wb for csp in self.csps[b]
            ], axis=-1)

        self.WT = np.moveaxis(self.W, -2, -1) # (B, M, C)
        self.WT = torch.tensor(self.WT, dtype=torch.float32, requires_grad=self.trainable)

    def forward(self, xfb: torch.Tensor) -> torch.Tensor:
        """Feature extraction using OVR-FBCSP algorithm

        Parameters
        ----------
        xfb : torch.Tensor
            EEG signals filtered by Filter Bank, shape (...B, C, T)
            B passbands, N trials, C channels, T discrete time

        Returns
        -------
        torch.Tensor
            Feature matrix, shape (..., B, M)
            with M = 2*m*nb_classes
        """
        assert self.WT is not None, "You should call self.fit first"
        xfb = xfb/sqrt(xfb.shape[-1]) # normalize by time length
        Z = self.WT @ xfb # (..., M, T)
        ZT = Z.moveaxis(-2, -1) # (..., T, M)
        ZZt = Z @ ZT # (..., M, M)
        diag = ZZt.diagonal(dim1=-2, dim2=-1) # (..., M)
        return torch.log(diag/diag.sum(-1, keepdim=True))

    # def transform(self, xfb: np.ndarray) -> np.ndarray:
    #     """Feature extraction using OVR-FBCSP algorithm

    #     Parameters
    #     ----------
    #     xfb : np.ndarray
    #         EEG signals filtered by Filter Bank, shape (B, N, C, T)
    #         B passbands, N trials, C channels, T discrete time

    #     Returns
    #     -------
    #     np.ndarray
    #         Feature matrix, shape (N, B, M)
    #         with M = 2*m*nb_classes
    #     """
    #     assert self.WT is not None, "You should call self.fit first"
    #     Z = self.WT @ xfb # (B, N, M, T)
    #     Z = np.moveaxis(Z, 0, 1) # (N, B, M, T)
    #     ZT = np.moveaxis(Z, -2, -1) # (N, B, T, M)
    #     ZZt = Z @ ZT # (N, B, M, M)
    #     diag = np.diagonal(ZZt, axis1=-2, axis2=-1)
    #     return np.log(diag/diag.sum(-1, keepdims=True))

