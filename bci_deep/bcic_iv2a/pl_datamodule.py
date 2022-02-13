"""
DataModule to feed PyTorch Lightning Trainer
"""
from copy import deepcopy
from typing import List, Optional, Tuple
from matplotlib.pyplot import get

import numpy as np
import pytorch_lightning as pl
import torch
from bci_deep.bcic_iv2a.torch_dataset import IV2aDataset
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import StratifiedShuffleSplit


class IV2aDataModule(pl.LightningDataModule):
    def __init__(self, data_dir,
                 nb_bands=9,
                 f_width=4, f_min=4, f_max=40,
                 gpass=3.0, gstop=30.0,
                 include_subject: List[str] = [],
                 exclude_subject: List[str] = [],
                 tmin=0.0, tmax=4.0,
                 train_transform=None, test_transform=None,
                 num_workers=2, batch_size=32, train_ratio=1.0, 
                 overwrite_sample=False,
                 bar_augmentation=False,
                 **kwargs):
        """DataModule to feed PyTorch Lightning Trainer
        This module wraps around IV2aDataset - a PyTorch Dataset subclass

        Parameters
        ----------
        data_dir : str
            Path to dataset directory
        nb_bands : int, optional
            Number of bands in Filter Bank, by default 9
        f_width : float, optional
            Bandwidth of passband filter in Filter Bank, in Hz
            By default 4
        f_min : float, optional
            Filter Bank lower limit in Hz, by default 4
        f_max : float, optional
            Filter Bank lower limit in Hz, by default 40
        gpass : float, optional
            The maximum loss in the passband (dB)
            By default 3.0
        gstop : float, optional
            The minimum attenuation in the stopband (dB)
            By default 30.0
        include_subject : List[str], optional
            List of subject that you want to get, ex: ["01", "02"]
            By default [], which mean all subjects
        exclude_subject : List[str], optional
            List of subject that you don't want to get, ex: ["01", "02"]
            By default []
        tmin, tmax: float
            Start and end time in seconds, relative to the start of each cue
            Defaults to 0.0 and 4.0 respectively (based on BCIC IV 2a description)
        train_transform, test_transform : optional
            Transformation to apply to EEG raw signals.
            Check bci_deep/bcic_iv2a/transform.py for inspiration
            By default None
        num_workers : int, optional
            Number of workers to fetch data, by default 2
        batch_size : int, optional
            By default 32
        train_ratio : float, optional
            Ratio to split "T" BCIC IV 2a dataset to train and validation.
            WARNING: this fucntion is not implemented.
            By default 1.0
        overwrite_sample : bool, optional
            If True, rebuild sample in npz format, by default False
        bar_augmentation : bool, optional
            If True, use Brain Area Recombination (BAR) in training.
            For details: https://www.frontiersin.org/articles/10.3389/fnhum.2021.645952/full
            By default False
        """
        super().__init__()
        self.dataset_kwargs = {
            "data_dir": data_dir,
            "nb_bands": nb_bands,
            "f_width": f_width,
            "f_min": f_min,
            "f_max": f_max,
            "gpass": gpass,
            "gstop": gstop,
            "include_subject": include_subject,
            "exclude_subject": exclude_subject,
            "tmin": tmin,
            "tmax": tmax,
        }
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.shuffle = True
        self.num_workers = num_workers
        self.trainset = None
        self.valset = None
        self.testset = None
        # decrease this factor if you have problem with RAM or CPU
        self.prefetch_factor = 16

    def split_train_val(self, dataset: IV2aDataset, train_ratio=0.8) -> Tuple[IV2aDataset]:
        """Not implemented yet
        """
        if train_ratio == 1.0:
            return dataset, None
        else:
            raise NotImplementedError

    def setup(self, stage: Optional[str] = None):
        """Setup DataModule

        Instantiate dataset and run setup with regard to the stage

        Parameters
        ----------
        stage : Optional[str], optional
            Either "fit" or "test"
            It is used to separate setup logic for trainer.{fit,validate,test}. 
            If None, all stages will be set up. By default None
        """
        if stage in (None, "fit"):
            dataset = IV2aDataset(**self.dataset_kwargs,
                                  train=True, transform=self.train_transform)
            dataset.setup()
            self.trainset, self.valset = self.split_train_val(
                dataset, self.train_ratio)
            self._has_setup_fit = True

        if stage in (None, "test"):
            self.dataset_kwargs["bar_augmentation"] = False
            self.testset = IV2aDataset(**self.dataset_kwargs,
                                       train=False, transform=self.test_transform)
            self.testset.setup()
            self._has_setup_test = True

    def train_dataloader(self):
        if getattr(self, "trainset", None) is None:
            raise ValueError("You should run `self.setup(stage='fit')`")
        return DataLoader(self.trainset, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.num_workers,
                          prefetch_factor=self.prefetch_factor)

    def val_dataloader(self):
        if getattr(self, "valset", None) is None:
            raise ValueError("You should run `self.setup(stage='fit')`")
        return DataLoader(self.valset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          prefetch_factor=self.prefetch_factor)

    def test_dataloader(self):
        if getattr(self, "testset", None) is None:
            raise ValueError("You should run `self.setup(stage='test')`")
        return DataLoader(self.testset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          prefetch_factor=self.prefetch_factor)
