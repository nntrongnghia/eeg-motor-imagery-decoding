"""
Hybrid Deep Neural Network from https://doi.org/10.1016/j.bspc.2020.102144
"""
from turtle import forward
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ml_collections import ConfigDict
from bci_deep.preprocess import FilterBank, OVR_CSP

def split_n_segments(x:torch.Tensor, nb_segments:int=4) -> torch.Tensor:
    """Split the signal in time. Use zero padding if needed.

    Parameters
    ----------
    x : torch.Tensor
        Shape (..., T)
    nb_segments : int, optional
        by default 4

    Returns
    -------
    torch.Tensor
        Splited signal, shape (L, ... splited T)
        with L number of segments
    """
    # Split to n sequences/segments using zero padding if needed
    pad_width = nb_segments - x.shape[-1] % nb_segments
    pad_before = pad_width // 2
    pad_after = pad_width - pad_before
    x = F.pad(x, (pad_before, pad_after))
    x = x.split(x.shape[-1]//nb_segments, -1) # list of (..., T/nb_segments)
    x = torch.stack(x) # (L, ..., seglen)
    return x


class Backbone(nn.Module):
    def __init__(self, input_dims: Tuple, cnn1_out_channels=4,
                 lstm_hidden_size=64, lstm_output_size=32,
                 lstm_input_size=32, lstm_num_layers=3, p_dropout=0.2) -> None:
        """Temporal and spatial feature extration using CNN and LSTM

        Parameters
        ----------
        input_dims : Tuple
            Input dimensions (L, 1, B, M)
            L: sequence length
            B: nb filter bands
            M: nb of features
        cnn1_out_channels : int, optional
            Output dimension of CNN1, by default 4
        lstm_hidden_size : int, optional
            LSTM cell hidden dimension, by default 64
        lstm_output_size : int, optional
            LSTM output dimension, by default 32
        lstm_input_size : int, optional
            LSTM input dimension, by default 32
        lstm_num_layers : int, optional
            Number of stacked LSTM cells, by default 3
        p_dropout : float, optional
            Probability of dropout, by default 0.2
            Note: not used in backbone
        """
        
        super().__init__()
        c1 = cnn1_out_channels  # c1 for short
        L, _, B, M = input_dims
        self.cnn1 = nn.Conv2d(1, c1, 5, padding="same")
        self.fc1 = nn.Sequential(
            nn.Linear(c1*B*M, lstm_input_size),
            nn.ReLU()
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(c1, c1*2, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(c1*2, c1*4, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(c1*4, c1*8, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            proj_size=lstm_output_size,
            num_layers=lstm_num_layers,
            batch_first=True)
        if lstm_output_size == 0:
            lstm_output_size = lstm_hidden_size

        cnn2_out_shape = (
            max(B // 8, 1), 
            max(M // 8, 1)
        )
        cnn2_out_res = cnn2_out_shape[0]*cnn2_out_shape[1]
        self.output_dims = L*(8*c1*cnn2_out_res + lstm_output_size)

            

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone forward

        Parameters
        ----------
        x : torch.Tensor
            EEG features from OVR-FBCSP
            shape (bs, L, 1, B, M)

        Returns
        -------
        torch.Tensor
            Combined features from CNN2 and LSTM
            shape (bs, L*(lstm_output_size + 8*cnn1_out_channels))
        """
        BS, L, _, B, M = x.shape
        cnn1_out = self.cnn1(x.reshape(-1, 1, B, M))
        spatial_ft = self.cnn2(cnn1_out)

        tokens = self.fc1(cnn1_out.reshape(BS, L, -1))
        temporal_ft, _ = self.lstm(tokens)

        spatial_ft = spatial_ft.reshape(BS, L, -1)
        combined_ft = torch.cat([spatial_ft, temporal_ft], dim=-1)
        return combined_ft.reshape(BS, -1)



class HDNN(nn.Module):
    def __init__(self, nb_segments=4,
                 m_filters=2, cnn1_out_channels=4, 
                 lstm_hidden_size=64, lstm_output_size=32,
                 lstm_input_size=32, lstm_num_layers=3,
                 p_dropout=0.2, head_hidden_dim=512, 
                 nb_classes=4, nb_bands=16,
                 trainable_csp=True, **kwargs) -> None:
        """Hybrid Deep Neural Network from https://doi.org/10.1016/j.bspc.2020.102144

        Parameters
        ----------
        nb_segments : int, optional
            Number of time segments to be splited, by default 4
        m_filters : int, optional
            Number of CSP features, by default 2
        cnn1_out_channels : int, optional
            Output dimension of CNN1, by default 4
        lstm_hidden_size : int, optional
            LSTM cell hidden dimension, by default 64
        lstm_output_size : int, optional
            LSTM output dimension, by default 32
        lstm_input_size : int, optional
            LSTM input dimension, by default 32
        lstm_num_layers : int, optional
            Number of stacked LSTM cells, by default 3
        p_dropout : float, optional
            Probability of dropout, by default 0.2
        head_hidden_dim : int, optional
            MLP classifier hidden dimension, by default 512
        nb_classes : int, optional
            Number of classes, by default 4
        
        """
        
        super().__init__()
        self.has_csp = True
        self.nb_segments = nb_segments
        input_dims = (nb_segments, 1, nb_bands, 2*m_filters*nb_classes)
        self.ovr_csp = OVR_CSP(nb_classes, m_filters, trainable_csp, nb_bands)
        self.backbone = Backbone(input_dims, cnn1_out_channels,
                                 lstm_hidden_size, lstm_output_size,
                                 lstm_input_size, lstm_num_layers, p_dropout)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.output_dims, head_hidden_dim),
            nn.ReLU(),

            nn.Linear(head_hidden_dim, head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p_dropout),

            nn.Linear(head_hidden_dim, 32),
            nn.ReLU(),
            
            nn.Linear(32, nb_classes),
        )
        self.initialize_weights()

    @torch.no_grad()
    def initialize_csp(self, xfb:np.ndarray, y:np.ndarray, on_gpu=False):
        """Initialize CSP transformation matrix

        Parameters
        ----------
        x : np.ndarray
            Filtered EEG signals, shape (B, N, C, T)
            B filter bands, N trials, C channels, T time
        y : np.ndarray
            labels, shape (N,)
        """
        self.ovr_csp.fit(xfb, y)

    def initialize_weights(self):
        """Initialize weights using Normal Distribution of mean 0 and std 0.1
        based on the paper
        """
        for param in self.parameters():
            torch.nn.init.trunc_normal_(param, std=0.1, a=-0.2, b=0.2)

    def finetune(self):
        """Freeze the backbone for transfer learning
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.head.parameters():
            torch.nn.init.trunc_normal_(param, std=0.1, a=-0.2, b=0.2)
        

    def forward(self, xfb: torch.Tensor, return_dict=False) -> torch.Tensor:
        """HDNN forward

        Parameters
        ----------
        x : torch.Tensor
            Filtered EEG signals, shape (N, B, C, T)
            N trials, B filter bands, C channels, T time

        Returns
        -------
        torch.Tensor
            classification softmax scores
            shape (N, nb_classes)
        """
        xfbs = split_n_segments(xfb, self.nb_segments).moveaxis(0, 1) # (N, L, B, C, T)
        csp_ft = self.ovr_csp(xfbs)
        x = self.backbone(csp_ft.unsqueeze(-3))
        logits = self.head(x)
        if return_dict:
            return {"logits": logits, "features":x}
        else:
            return logits
