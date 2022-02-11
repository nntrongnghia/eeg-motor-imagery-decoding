from math import sqrt
import numpy as np
import torch
import torch.nn as nn
from bci_deep.preprocess.filter_bank import FilterBank
from bci_deep.preprocess.csp import CSP


class OVR_CSP(nn.Module):
    def __init__(self, nb_classes: int, m_filters=2, trainable=True, nb_bands=16, nb_channels=22):
        """One Versus Rest - Common Spatial Pattern

        Parameters
        ----------
        nb_classes: int
            Number of classes
        m_filters : int, optional
            Number of CSP filters, by default 2
        nb_bands: int, optional
            Number of Filter Bank passbands
        """
        super().__init__()
        self.trainable = trainable
        # CSP for each class of each filter band
        self.csps = None
        self.nb_classes = nb_classes
        self.m_filters = m_filters
        self.nb_bands = nb_bands
        self.nb_channels = nb_channels
        
        # Transformation matrix (B, M, C)
        # with M = m_filters*2*nb_classes
        # Transposed W with expanded dims, ready to use in self.transform
        WT = torch.zeros(
            (nb_bands, 2*m_filters*nb_classes, nb_channels), 
            dtype=torch.float32)
        torch.nn.init.normal_(WT, std=0.1)
        self.WT = nn.parameter.Parameter(WT, requires_grad=trainable)

    @torch.no_grad()
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
        B = self.nb_bands
        C = self.nb_channels
        classes = np.unique(y_labels)
        assert B == x_fb.shape[0], f"B={B}, x_fb.shape[0]={x_fb.shape[0]}"
        assert C == x_fb.shape[2], f"C={B}, x_fb.shape[2]={x_fb.shape[2]}"
        assert len(classes) == self.nb_classes, \
            f"y_labels should contain {self.nb_classes} classes"
        self.csps = [[CSP(self.m_filters) for _ in range(self.nb_classes)] for _ in range(B)]
        W = np.zeros((B, C, self.m_filters*2*self.nb_classes))
        # TODO: optimize code to remove for loops ?
        for b in range(B):
            x = x_fb[b]  # (N, C, T)
            for cls in classes:
                y_ovr = (y_labels == cls).astype(np.int)
                self.csps[b][cls].fit(x, y_ovr)
            W[b] = np.concatenate([csp.Wb for csp in self.csps[b]], axis=-1)

        WT = np.moveaxis(W, -2, -1) # (B, M, C)
        WT = torch.tensor(WT, dtype=torch.float32)
        self.WT.copy_(WT)

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
