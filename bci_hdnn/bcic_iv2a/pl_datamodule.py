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
    def __init__(self, data_dir,
                 nb_segments=4, nb_bands=9,
                 f_width=4, f_min=4, f_max=40,
                 gpass=3.0, gstop=30.0,
                 include_subject: List[str] = [],
                 exclude_subject: List[str] = [],
                 tmin=0.0, tmax=4.0,
                 train_transform=None, test_transform=None,
                 num_workers=2, batch_size=32, train_ratio=0.8, 
                 overwrite_sample=False,
                 **kwargs):
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
            "overwrite_sample": overwrite_sample
        }
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.shuffle = True
        self.num_workers = num_workers

    def split_train_val(self, dataset: IV2aDataset, train_ratio=0.8) -> Tuple[IV2aDataset]:
        if train_ratio == 1.0:
            return dataset, None
        else:
            raise NotImplementedError
        # spliter = StratifiedShuffleSplit(1, train_size=train_ratio)
        # y = dataset.y + dataset.s*10
        # train_idx, val_idx = next(spliter.split(dataset.x, y))
        # val_dataset = deepcopy(dataset)
        # train_dataset = dataset

        # val_dataset.x = val_dataset.x[val_idx]
        # val_dataset.y = val_dataset.y[val_idx]
        # val_dataset.s = val_dataset.s[val_idx]
        # val_dataset.transform = self.test_transform

        # train_dataset.x = train_dataset.x[train_idx]
        # train_dataset.y = train_dataset.y[train_idx]
        # train_dataset.s = train_dataset.s[train_idx]

        # return train_dataset, val_dataset

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
            dataset = IV2aDataset(**self.dataset_kwargs,
                                  train=True, transform=self.train_transform)
            dataset.setup()
            self.trainset, self.valset = self.split_train_val(
                dataset, self.train_ratio)
            self._has_setup_fit = True

        if stage in (None, "test"):
            self.testset = IV2aDataset(**self.dataset_kwargs,
                                       train=False, transform=self.test_transform)
            self.testset.setup()
            self._has_setup_test = True

    def train_dataloader(self):
        if getattr(self, "trainset", None) is None:
            raise ValueError("You should run `self.setup(stage='fit')`")
        return DataLoader(self.trainset, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.num_workers,
                          prefetch_factor=16)

    def val_dataloader(self):
        if getattr(self, "valset", None) is None:
            raise ValueError("You should run `self.setup(stage='fit')`")
        return DataLoader(self.valset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          prefetch_factor=16)

    def test_dataloader(self):
        if getattr(self, "testset", None) is None:
            raise ValueError("You should run `self.setup(stage='test')`")
        return DataLoader(self.testset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          prefetch_factor=16)


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
