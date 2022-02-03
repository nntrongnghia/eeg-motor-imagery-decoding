from copy import deepcopy
from typing import List, Optional, Tuple
from matplotlib.pyplot import get

import numpy as np
import pytorch_lightning as pl
import torch
from bci_hdnn.bcic_iv2a.torch_dataset import IV2aDataset
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import StratifiedShuffleSplit    


class IV2aDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, train_ratio=0.8, nb_segments=4, nb_bands=9,
                 m_filters=2, include_subject: List[str] = [], exclude_subject: List[str] = [],
                 tmin=0.0, tmax=4.0, time_window=None,
                 train_transform=None, test_transform=None, 
                 num_workers=2, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.nb_segments = nb_segments
        self.nb_bands = nb_bands
        self.m_filters = m_filters
        self.tmin, self.tmax = tmin, tmax
        self.include_subjects = include_subject
        self.exclude_subjects = exclude_subject
        self.train_transforms = train_transform
        self.test_transforms = test_transform
        self.time_window = time_window
        self.preprocessors = None
        self.shuffle = True
        self.num_workers = num_workers
        self.dims = None

    def split_train_val(self, dataset:IV2aDataset, train_ratio=0.8) -> Tuple[IV2aDataset]:
        if train_ratio == 1.0:
            return dataset, None
        spliter = StratifiedShuffleSplit(1, train_size=train_ratio)
        y = dataset.y + dataset.s*10
        train_idx, val_idx = next(spliter.split(dataset.x, y))
        val_dataset = deepcopy(dataset)
        train_dataset = dataset

        val_dataset.x = val_dataset.x[val_idx]
        val_dataset.y = val_dataset.y[val_idx]
        val_dataset.s = val_dataset.s[val_idx]
        val_dataset.transform = self.test_transforms

        train_dataset.x = train_dataset.x[train_idx]
        train_dataset.y = train_dataset.y[train_idx]
        train_dataset.s = train_dataset.s[train_idx]

        return train_dataset, val_dataset


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
            dataset = IV2aDataset(self.data_dir, self.nb_segments, True, 
                self.include_subjects, self.exclude_subjects, self.tmin, self.tmax, 
                self.time_window, self.train_transforms, self.nb_bands)
            dataset.setup()
            self.preprocessors = deepcopy(dataset.preprocessors)
            self.dims = deepcopy(dataset.dims)
            self.trainset, self.valset = self.split_train_val(dataset, self.train_ratio)
            self._has_setup_fit = True

        if stage in (None, "test"):
            if self.preprocessors is None:
                train_dataset = IV2aDataset(self.data_dir, self.nb_segments, True, 
                    self.include_subjects, self.exclude_subjects, self.tmin, self.tmax, 
                    self.time_window, self.train_transforms, self.nb_bands)
                train_dataset.setup()
                self.preprocessors = deepcopy(dataset.preprocessors)
                del train_dataset

            self.testset = IV2aDataset(self.data_dir, self.nb_segments, False, 
                self.include_subjects, self.exclude_subjects, self.tmin, self.tmax, 
                self.time_window, self.test_transforms, self.nb_bands)
            self.testset.load_external_preprocessors(self.preprocessors)
            self.testset.setup()
            self._has_setup_test = True

    def train_dataloader(self):
        if getattr(self, "trainset", None) is None:
            raise ValueError("You should run `self.setup(stage='fit')`")
        return DataLoader(self.trainset, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.num_workers)

    def val_dataloader(self):
        if getattr(self, "valset", None) is None:
            raise ValueError("You should run `self.setup(stage='fit')`")
        return DataLoader(self.valset, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.num_workers)

    def test_dataloader(self):
        if getattr(self, "testset", None) is None:
            raise ValueError("You should run `self.setup(stage='test')`")
        return DataLoader(self.testset, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.num_workers)


# test code
if __name__ == "__main__":
    import pytorch_lightning as pl

    # for reproducibility
    pl.seed_everything(42, workers=True)
    from bci_hdnn.model.config import hdnn_base_config

    config = hdnn_base_config()
    data_dir = "/home/nghia/dataset/BCI_IV_2a"
    datamodule = IV2aDataModule(data_dir, exclude_subject=["01"], **config)
    datamodule.setup(stage="fit")
    print("Done")
