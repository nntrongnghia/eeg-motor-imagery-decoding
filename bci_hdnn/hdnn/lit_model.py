from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from bci_hdnn.hdnn import HDNN
from torchmetrics import Accuracy, CohenKappa, ConfusionMatrix


class LitModel(pl.LightningModule):
    def __init__(self, model: nn.Module, nb_classes=4, lr=0.001, **kwargs) -> None:
        super().__init__()
        self.lr = lr
        self.model = model
        self.kappa = CohenKappa(nb_classes)
        # self.confusion = ConfusionMatrix(nb_classes)
        self.accuracy = Accuracy()
        self.nb_classes = nb_classes


    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.001)
        return torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch["ft"], batch["y"].reshape(-1,)
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
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
        # self.confusion(ypred, y)
        self.accuracy(ypred, y)
        return loss

    def log_confusion_matrix(self):
        confusion_matrix = self.confusion.compute().cpu().numpy()
        df_cm = pd.DataFrame(
            confusion_matrix, 
            index=range(self.nb_classes), 
            columns=range(self.nb_classes))
        plt.figure(figsize = (10,7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
        plt.close(fig_)
        self.logger.experiment.add_figure("Confusion matrix", fig_, self.current_epoch)

    def validation_epoch_end(self, outputs):
        # self.log_confusion_matrix()
        # self.confusion.reset()
        self.log("val_kappa", self.kappa.compute())
        self.log("val_accuracy", self.accuracy.compute())
        self.kappa.reset()
        self.accuracy.reset()

        


    