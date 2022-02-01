from turtle import forward
from typing import Tuple
import torch
import torch.nn as nn
from ml_collections import ConfigDict

class Backbone(nn.Module):
    def __init__(self, input_dims: Tuple, cnn1_out_channels=4, 
                 lstm_hidden_size=64, lstm_output_size=32,
                 lstm_input_size=32, lstm_num_layers=3) -> None:
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
        c1 = cnn1_out_channels # c1 for short
        L, _, B, M = input_dims
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, c1, 5, padding="same"),
            nn.ReLU()
        )
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
            nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pool
        )
        self.lstm = nn.LSTM(
            input_size=lstm_input_size, 
            hidden_size=lstm_hidden_size, 
            proj_size=lstm_output_size,
            num_layers=lstm_num_layers,
            batch_first=True)
        self.output_dims = L*(8*c1 + lstm_output_size)

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
    def __init__(self, config: ConfigDict) -> None:
        super().__init__()
        self.backbone = Backbone(**config)
        self.head = nn.Sequential(
            nn.Linear(self.backbone.output_dims, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(config.p_dropout),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, config.nb_classes),
            nn.Softmax()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        scores = self.head(x)
        return scores

# test 
if __name__ == "__main__":
    backbone = Backbone((4, 1, 9, 16))
    x = torch.rand(32, 4, 1, 9, 16)
    ft = backbone(x)
    print(ft.shape)