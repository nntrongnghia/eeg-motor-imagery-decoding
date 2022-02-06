import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, nb_classes=4) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 32),
            nn.LeakyReLU(),
            nn.Linear(32, nb_classes)
        )
        self.input_dim = input_dim
    
    def forward(self,x):
        return self.mlp(x.reshape(-1, self.input_dim))