# TODO: data augmentation for EEG ?
import logging
from typing import Dict, List
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from bci_hdnn.bcic_iv2a import BCIC_IV2a
from bci_hdnn.preprocess import OVR_CSP, FilterBank
from torch.utils.data import Dataset
from bci_hdnn.bcic_iv2a.transform import ToTensor


class IV2aDataset(Dataset):
    NB_CLASSES = 4
    FS = 250
    def __init__(self,
                 data_dir, train: bool = True, nb_segments=4,
                 nb_bands=9, f_width=4, f_min=4, f_max=40, gpass=3, gstop=30,
                 include_subject: List[str] = [], 
                 exclude_subject: List[str] = [],
                 tmin=0.0, tmax=4.0, transform=None) -> None:
        super().__init__()
        self.filter = FilterBank(self.FS, nb_bands, f_width, f_min, f_max, gpass, gstop)
        self.nb_segments = nb_segments
        self.tmin, self.tmax = tmin, tmax
        self.train = train
        self.include_subjects = include_subject
        self.exclude_subjects = exclude_subject
        self.dataset = BCIC_IV2a(data_dir)
        self.x, self.y, self.s = None, None, None
        self.transform = transform


    def build_subject_list(self):
        self.subject_list = []
        for filename in self.dataset.filenames:
            s = filename[1:3]
            # Only include subjects in include_subjects
            # if this list is not empty
            if len(self.include_subjects) > 0:
                if s not in self.include_subjects:
                    continue
            # Exclude subjects in exclude_subjects
            # if this list is not empty
            if len(self.exclude_subjects) > 0:
                if s in self.exclude_subjects:
                    continue
            if s not in self.subject_list:
                self.subject_list.append(s)
        self.subject_list = [int(s) for s in self.subject_list]

    def load_data(self):
        data = []
        if self.train:
            self.preprocessors = {s: None for s in self.subject_list}
            filenames = [name for name in self.dataset.filenames if "T" in name]
        else:
            filenames = [name for name in self.dataset.filenames if "E" in name]
        for name in filenames:
            subject = int(name[1:3])
            if subject not in self.subject_list:
                continue
            subject_data = self.dataset.read_file(name, self.tmin, self.tmax)
            nb_trials = subject_data["x_data"].shape[0]
            subject_data["subject"] = np.array([int(subject_data["subject"])]*nb_trials)
            data.append(subject_data)

        self.x = np.concatenate([d["x_data"] for d in data]) # (N, C, T)
        self.y = np.concatenate([d["y_labels"] for d in data]) # (N, )
        self.s = np.concatenate([d["subject"] for d in data]) # (N, )


    def setup(self):
        logging.info("IV2aDataset setup ...")
        self.build_subject_list()
        self.load_data()
        

    def clean_memory(self):
        """Clean arrays to reduce memory footprint
        """
        logging.info("IV2aDataset clean arrays")
        if self.x is not None:
            del self.x
            self.x = None
        if self.y is not None:
            del self.y
            self.y = None
        if self.s is not None:
            del self.s
            self.s = None

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        assert self.x is not None, "self.setup should be run first"
        x = self.x[idx]  # (C, T)
        y = self.y[idx].reshape((-1, ))
        s = self.s[idx].reshape((-1, ))

        if self.transform is not None:
            x = self.transform(x)
        
        xfb = self.filter(torch.tensor(x)).moveaxis(-2, -3) # (B, C, T)

        sample = {
            "y": y,
            "s": s
        }
        sample = ToTensor()(sample)
        sample["xfb"] = xfb.to(torch.float32)
        return sample


# test code
if __name__ == "__main__":
    from sklearn.model_selection import StratifiedShuffleSplit   
    logging.getLogger().setLevel(logging.INFO)
    dataset = IV2aDataset("/home/nghia/dataset/BCI_IV_2a")
    dataset.setup()
    print(len(dataset))
    spliter = StratifiedShuffleSplit(1, train_size=0.8)
    y = dataset.y + dataset.s*10
    train_idx, val_idx = next(spliter.split(dataset.x, y))
    print("Train split", np.unique(y[train_idx], return_counts=True))
    print("Val split", np.unique(y[val_idx], return_counts=True))
    print("Done")
