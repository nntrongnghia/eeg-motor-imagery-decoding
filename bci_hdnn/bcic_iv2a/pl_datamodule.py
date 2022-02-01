from copy import deepcopy
from typing import List, Optional
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from bci_hdnn.bcic_iv2a.torch_dataset import IV2aDataset


class IV2aDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, train_ratio=0.8, nb_segments=4,
                 include_subject: List[str] = [], exclude_subject: List[str] = [],
                 tmin=0.0, tmax=4.0, train_transform=None, test_transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.nb_segments = nb_segments
        self.tmin, self.tmax = tmin, tmax
        self.include_subjects = include_subject
        self.exclude_subjects = exclude_subject
        self.train_transforms = train_transform
        self.test_transforms = test_transform
        self.preprocessors = None
        self.shuffle = True
        self.num_workers = 2
        self.dims = None

    def setup(self, stage: Optional[str] = None):
        """Setup DataModule

        1. Instantiate dataset w.r.t. `stage`
        2. Preload EEG samples with corresponding labels
        3. Fit OVR-FBCSP

        Parameters
        ----------
        stage : Optional[str], optional
            Either "fit" or "test"
            It is used to separate setup logic for trainer.{fit,validate,test}. 
            If None, all stages will be set up. By default None
        """
        if stage in (None, "fit"):
            dataset = IV2aDataset(
                self.data_dir, self.nb_segments, True, self.include_subjects,
                self.exclude_subjects, self.tmin, self.tmax, self.train_transforms)
            dataset.setup()
            self.preprocessors = deepcopy(dataset.preprocessors)
            self.dims = deepcopy(dataset.dims)
            train_len = int(len(dataset)*self.train_ratio)
            val_len = len(dataset) - train_len
            self.trainset, self.valset = random_split(
                dataset, [train_len, val_len])
            self._has_setup_fit = True

        if stage in (None, "test"):
            if self.preprocessors is None:
                train_dataset = IV2aDataset(
                    self.data_dir, self.nb_segments, True, self.include_subjects,
                    self.exclude_subjects, self.tmin, self.tmax, self.train_transforms)
                train_dataset.setup()
                self.preprocessors = deepcopy(dataset.preprocessors)
                del train_dataset

            self.testset = IV2aDataset(
                self.data_dir, self.nb_segments, False, self.include_subjects,
                self.exclude_subjects, self.tmin, self.tmax, self.test_transforms)
            self.testset.setup()
            self.testset.load_external_preprocessors(self.preprocessors)
            self._has_setup_test = True

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.num_workers)