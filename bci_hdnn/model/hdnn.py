from turtle import forward
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ml_collections import ConfigDict


class Backbone(nn.Module):
    def __init__(self, input_dims: Tuple, cnn1_out_channels=4,
                 lstm_hidden_size=64, lstm_output_size=32,
                 lstm_input_size=32, lstm_num_layers=3, p_dropout=0.2) -> None:
        """Temporal and spatial feature extration

        Parameters
        ----------
        input_dims : Tuple[int, int, int, int]
            Input dimensions (L, 1, B, M)
            L: sequence length
            B: nb filter bands
            M: nb of features
        """
        super().__init__()
        c1 = cnn1_out_channels  # c1 for short
        L, _, B, M = input_dims
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, c1, 5, padding="same"),
            nn.LeakyReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(c1*B*M, lstm_input_size),
            nn.LeakyReLU()
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(c1, c1*2, 3, padding="same"),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(c1*2, c1*4, 3, padding="same"),
            # nn.Dropout2d(p_dropout),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(c1*4, c1*8, 3, padding="same"),
            # nn.Dropout2d(p_dropout),
            nn.LeakyReLU(),
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
    def __init__(self, input_dims: Tuple, cnn1_out_channels=4,
                 lstm_hidden_size=64, lstm_output_size=32,
                 lstm_input_size=32, lstm_num_layers=3,
                 p_dropout=0.2, head_hidden_dim=512, nb_classes=4, **kwargs) -> None:
        super().__init__()
        self.backbone = Backbone(input_dims, cnn1_out_channels,
                                 lstm_hidden_size, lstm_output_size,
                                 lstm_input_size, lstm_num_layers, p_dropout)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.output_dims, head_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(head_hidden_dim, head_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(head_hidden_dim, 32),
            nn.LeakyReLU(),
            nn.Linear(32, nb_classes),
        )
        # self.initialize_weights()

    def initialize_weights(self):
        for param in self.parameters():
            nn.init.normal_(param, mean=0, std=0.1)


    def finetune(self):
        for param in self.backbone.parameters():
            param.requires_grad = False


    def forward(self, x: torch.Tensor, return_score=False) -> torch.Tensor:
        """HDNN forward

        Parameters
        ----------
        x : torch.Tensor
            EEG features from OVR-FBCSP
            shape (bs, L, 1, B, M)

        Returns
        -------
        torch.Tensor
            classification softmax scores
            shape (bs, nb_classes)
        """
        x = self.backbone(x)
        logits = self.head(x)
        if return_score:
            return torch.softmax(logits, dim=-1)
        else:
            return logits




# test
if __name__ == "__main__":
    backbone = Backbone((4, 1, 9, 16))
    x = torch.rand(32, 4, 1, 9, 16)
    ft = backbone(x)
    print(ft.shape)
