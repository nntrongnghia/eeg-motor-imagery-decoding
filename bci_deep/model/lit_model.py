import logging
from typing import Dict, Tuple
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
from bci_deep.model.losses import ce_loss



class LitModel(pl.LightningModule):
    def __init__(self, model_class: nn.Module=HDNN,
                 input_key="eeg_fb",
                 nb_classes=4,
                 lr=0.001,
                 loss_fn=ce_loss,
                 **model_kwargs) -> None:
        """A Lightning Module warps the model and train/val/test steps

        Parameters
        ----------
        model_class : nn.Module
            Subclass of nn.Module, use to build the model.
            The model input: filtered EEG signals of shape (N, C, B, T)
            N trials, C channels, B filter bands, T time
        model_kwargs : dict, optional
            Keyword arguments to instantiate the model
            By default {}
        nb_classes : int, optional
            Number of classes, by default 4
        lr : float, optional
            Learning rate, by default 0.001
        loss_fn: function
            Signature: loss_fn(y_pred, y_target)
            Return a torch scalar for loss
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.model_class = model_class
        model_kwargs["nb_classes"] = nb_classes
        self.model = model_class(**model_kwargs)
        self.nb_classes = nb_classes
        self.criterion = loss_fn
        self.input_key = input_key

        # Metrics
        self.max_val_kappa = 0
        self.train_kappa = CohenKappa(nb_classes)
        self.train_accuracy = Accuracy()

        self.kappa = CohenKappa(nb_classes)
        self.accuracy = Accuracy()

        self.confusion = ConfusionMatrix(nb_classes)

    @torch.no_grad()
    def initialize_csp(self, datamodule: pl.LightningDataModule):
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
        if self.model.has_csp:
            dataset = datamodule.trainset
            dataset.transform = datamodule.test_transforms
            
            B, C, T = dataset[0]["eeg_fb"].shape

            if len(dataset) > 512:
                sample_idx = np.random.randint(0, len(dataset), 512)
            else:
                sample_idx = list(range(len(dataset)))

            xfb = []
            y = []
            for idx in sample_idx:
                sample = dataset[idx]
                xfb.append(sample["eeg_fb"])
                y.append(sample["y"])

            xfb = torch.cat(xfb).reshape(-1, B, C, T).moveaxis(1, 0).cpu().numpy() # (B, N, C, T)
            y = torch.stack(y).cpu().numpy()
            self.model.initialize_csp(xfb, y)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def finetune(self):
        """Wraper for model.fintune() 
        which freezes some layers for transfer learning (finetuning)
        """
        self.model.finetune()

    def forward(self, x):
        return self.model(x, return_dict=True)

    def predict_step(self, x):
        m_outputs = self(x)
        logits = m_outputs["logits"]
        ypred = torch.argmax(logits, -1)
        return ypred

    def training_step(self, batch, batch_idx):
        x, y = batch[self.input_key], batch["y"].reshape(-1,)
        # inference
        m_outputs = self(x)
        logits = m_outputs["logits"]
        pred_scores = torch.softmax(logits, -1)
        ypred = torch.argmax(pred_scores, -1)
        # loss
        loss = self.criterion(m_outputs, y)
        self.log("train_loss", loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        # metrics
        self.train_kappa(ypred, y)
        self.train_accuracy(ypred, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[self.input_key], batch["y"].reshape(-1,)
        # inference
        m_outputs = self(x)
        logits = m_outputs["logits"]
        pred_scores = torch.softmax(logits, -1)
        ypred = torch.argmax(pred_scores, -1)
        # loss
        loss = self.criterion(m_outputs, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        # metrics
        self.kappa(ypred, y)
        self.accuracy(ypred, y)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch[self.input_key], batch["y"].reshape(-1,)
        # inference
        m_outputs = self(x)
        logits = m_outputs["logits"]
        pred_scores = torch.softmax(logits, -1)
        ypred = torch.argmax(pred_scores, -1)
        # loss
        loss = self.criterion(m_outputs, y)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        # metrics
        self.kappa(ypred, y)
        self.accuracy(ypred, y)
        self.confusion(ypred, y)
        return loss

    def log_confusion_matrix(self, confusion_matrix_metric: ConfusionMatrix):
        confusion_matrix = confusion_matrix_metric.compute().cpu().numpy()
        df_cm = pd.DataFrame(
            confusion_matrix,
            index=range(self.nb_classes),
            columns=range(self.nb_classes))
        print("CONFUSION MATRIX")
        print(df_cm)
        if self.logger is not None:
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
        self.log("train_kappa", self.train_kappa.compute())
        self.log("train_accuracy", self.train_accuracy.compute())
        if self.trainer._progress_bar_callback is None:
            loss = float(torch.stack(
                [out["loss"] for out in outputs]
            ).mean().detach().cpu().numpy())
            acc = float(self.train_accuracy.compute().cpu().numpy())
            kappa = float(self.train_kappa.compute().cpu().numpy())
            logging.info(
                f"Epoch {self.current_epoch}, train_loss={loss:.3f}, train_kappa={kappa:.3f}, train_acc={acc:.3f}")
        self.train_kappa.reset()
        self.train_accuracy.reset()

    def validation_epoch_end(self, outputs):
        self.log("val_kappa", self.kappa.compute())
        self.log("val_accuracy", self.accuracy.compute())
        kappa = float(self.kappa.compute().cpu().numpy())
        if kappa > self.max_val_kappa:
            self.max_val_kappa = kappa
        self.log("max_val_kappa", self.max_val_kappa)
        
        if self.trainer._progress_bar_callback is None:
            loss = float(torch.stack(outputs).mean().detach().cpu().numpy())
            acc = float(self.accuracy.compute().cpu().numpy())
            logging.info(
                f"Epoch {self.current_epoch}, val_loss={loss:.3f}, val_kappa={kappa:.3f}, val_acc={acc:.3f}")
        
        self.kappa.reset()
        self.accuracy.reset()

    def test_epoch_end(self, outputs) -> None:
        self.log_confusion_matrix(self.confusion)
        self.log("test_kappa", self.kappa.compute())
        self.log("test_accuracy", self.accuracy.compute())
        self.kappa.reset()
        self.accuracy.reset()
        self.confusion.reset()
