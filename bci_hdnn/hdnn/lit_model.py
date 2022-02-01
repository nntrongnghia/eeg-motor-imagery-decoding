from typing import Tuple
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import CohenKappa, ConfusionMatrix, AveragePrecision
from bci_hdnn.hdnn import HDNN


class LitHDNN(pl.LightningModule):
    def __init__(self, input_dims: Tuple, cnn1_out_channels=4,
                 lstm_hidden_size=64, lstm_output_size=32,
                 lstm_input_size=32, lstm_num_layers=3,
                 p_dropout=0.2, nb_classes=4, lr=0.001, **kwargs) -> None:
        super().__init__()
        self.lr = lr
        self.model = HDNN(input_dims, cnn1_out_channels,
                          lstm_hidden_size, lstm_output_size,
                          lstm_input_size, lstm_num_layers,
                          p_dropout, nb_classes)
        self.kappa = CohenKappa(nb_classes)
        self.confusion = ConfusionMatrix(nb_classes)


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def forward(self, x, return_score=False):
        return self.model(x, return_score)
    
    def training_step(self, batch, batch_idx):
        x, y = batch["ft"], batch["y"].reshape(-1,)
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["ft"], batch["y"].reshape(-1,)
        # inference
        logits = self(x)
        pred_scores = torch.softmax(logits, -1)
        ypred = torch.argmax(pred_scores, -1)
        # loss
        loss = F.cross_entropy(logits, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        # metrics
        self.kappa(ypred, y)
        self.confusion(ypred, y)
        return loss

    def validation_epoch_end(self, outputs):
        self.log("val_consufion", self.confusion.compute())
        self.log("val_kappa", self.kappa.compute())

        self.confusion.reset()
        self.kappa.reset()

        


    