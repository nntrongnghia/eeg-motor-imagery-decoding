from inspect import CO_ASYNC_GENERATOR
from random import sample
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from bci_deep.model import HDNN
from torchmetrics import Accuracy, CohenKappa, ConfusionMatrix


class LitModel(pl.LightningModule):
    def __init__(self, model_class: nn.Module = None, 
                model_kwargs={}, 
                nb_classes=4, 
                lr=0.001, 
                cls_weights=None,
                **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.model = model_class(**model_kwargs)
        self.nb_classes = nb_classes
        if cls_weights is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.CrossEntropyLoss(cls_weights)

        # Train metrics
        self.train_kappa = CohenKappa(nb_classes)
        self.train_accuracy = Accuracy()
        # self.train_confusion = ConfusionMatrix(nb_classes)

        # Validation metrics
        self.val_kappa = CohenKappa(nb_classes)
        self.val_accuracy = Accuracy()
        # self.val_confusion = ConfusionMatrix(nb_classes)

        # Test metrics
        self.test_kappa = CohenKappa(nb_classes)
        self.test_accuracy = Accuracy()
        self.test_confusion = ConfusionMatrix(nb_classes)

    @torch.no_grad()
    def initialize_csp(self, train_dataloader: DataLoader):
        """Initialize CSP transformation matrix

        Parameters
        ----------
        train_dataloader: DataLoader
            Generate xfb, y, in which
            x : np.ndarray
                Filtered EEG signals, shape (Bs, B, C, T)
                B filter bands, L segments, C channels, T time
            y : np.ndarray
                labels, shape (N,)
        """
        dataset = train_dataloader.dataset
        B, C, T = dataset[0]["xfb"].shape

        if len(dataset) > 512:
            sample_idx = np.random.randint(0, len(dataset), 512)
        else:
            sample_idx = list(range(len(dataset)))

        xfb = []
        y = []
        for idx in sample_idx:
            sample = dataset[idx]
            xfb.append(sample["xfb"])
            y.append(sample["y"])

        xfb = torch.cat(xfb).reshape(-1, B, C, T).moveaxis(1, 0).cpu().numpy()
        y = torch.stack(y).cpu().numpy()
        self.model.initialize_csp(xfb, y)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": scheduler,
        #     "inverval": "epoch",
        #     "frequency": 1
        # }

    def finetune(self):
        self.model.finetune()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["xfb"], batch["y"].reshape(-1,)
        # inference
        logits = self(x)
        pred_scores = torch.softmax(logits, -1)
        ypred = torch.argmax(pred_scores, -1)
        # loss
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        # metrics
        self.train_kappa(ypred, y)
        self.train_accuracy(ypred, y)
        # self.train_confusion(ypred, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["xfb"], batch["y"].reshape(-1,)
        # inference
        logits = self(x)
        pred_scores = torch.softmax(logits, -1)
        ypred = torch.argmax(pred_scores, -1)
        # loss
        loss = self.criterion(logits, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        # metrics
        self.val_kappa(ypred, y)
        self.val_accuracy(ypred, y)
        # self.val_confusion(ypred, y)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch["xfb"], batch["y"].reshape(-1,)
        # inference
        logits = self(x)
        pred_scores = torch.softmax(logits, -1)
        ypred = torch.argmax(pred_scores, -1)
        # loss
        loss = self.criterion(logits, y)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        # metrics
        self.test_kappa(ypred, y)
        self.test_accuracy(ypred, y)
        self.test_confusion(ypred, y)
        return loss

    def log_confusion_matrix(self, confusion_matrix_metric: ConfusionMatrix):
        confusion_matrix = confusion_matrix_metric.compute().cpu().numpy()
        df_cm = pd.DataFrame(
            confusion_matrix,
            index=range(self.nb_classes),
            columns=range(self.nb_classes))
        plt.figure(figsize=(10, 7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
        plt.xlabel("Predictions")
        plt.ylabel("Targets")
        plt.close(fig_)
        self.logger.experiment.add_figure(
            "Confusion matrix", fig_, self.current_epoch)

        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()
        del fig_
        del df_cm

    def training_epoch_end(self, outputs) -> None:
        # self.log_confusion_matrix(self.train_confusion)
        self.log("train_kappa", self.train_kappa.compute())
        self.log("train_accuracy", self.train_accuracy.compute())
        self.train_kappa.reset()
        self.train_accuracy.reset()
        # self.train_confusion.reset()

    def validation_epoch_end(self, outputs):
        # self.log_confusion_matrix(self.val_confusion)
        self.log("val_kappa", self.val_kappa.compute())
        self.log("val_accuracy", self.val_accuracy.compute())
        self.val_kappa.reset()
        self.val_accuracy.reset()
        # self.val_confusion.reset()

    def test_epoch_end(self, outputs) -> None:
        self.log_confusion_matrix(self.test_confusion)
        self.log("test_kappa", self.test_kappa.compute())
        self.log("test_accuracy", self.test_accuracy.compute())
        self.test_kappa.reset()
        self.test_accuracy.reset()
        self.test_confusion.reset()
