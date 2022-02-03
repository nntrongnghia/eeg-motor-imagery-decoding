# TODO: data augmentation for EEG ?
import logging
from typing import Dict, List
from copy import deepcopy
import numpy as np
import torch
from bci_hdnn.bcic_iv2a import BCIC_IV2a
from bci_hdnn.preprocess import OVR_CSP, FilterBank
from torch.utils.data import Dataset
from bci_hdnn.bcic_iv2a.transform import ToTensor


class IV2aDataset(Dataset):
    NB_CLASSES = 4

    def __init__(self,
                 data_dir, nb_segments=4, train: bool = True,
                 include_subject: List[str] = [], exclude_subject: List[str] = [],
                 tmin=0.0, tmax=4.0, time_window=None,
                 transform=None, nb_bands=9, m_filters=2) -> None:
        super().__init__()
        self.nb_segments = nb_segments
        self.m_filters = m_filters
        self.nb_bands = nb_bands
        self.tmin, self.tmax = tmin, tmax
        self.time_window = time_window
        self.train = train
        self.include_subjects = include_subject
        self.exclude_subjects = exclude_subject
        self.dataset = BCIC_IV2a(data_dir)
        self.x, self.y, self.s = None, None, None
        self.dims = None
        self.transform = transform
        # self.setup()

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

    def load_data_and_preprocessors(self):
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
            fs = subject_data["fs"]
            if self.train:
                self.initialize_preprocessor(subject, subject_data, fs)

        self.x = np.concatenate([d["x_data"] for d in data]) # (N, C, T)
        self.y = np.concatenate([d["y_labels"] for d in data]) # (N, )
        self.s = np.concatenate([d["subject"] for d in data]) # (N, )

        # Get input dimensions
        self.get_ft_dims()

    def get_ft_dims(self):
        if self.preprocessors[self.subject_list[0]] is None:
            raise ValueError("You should setup dataset in `train=True` or use self.load_external_preprocessors")
        s = self.subject_list[0]
        xfb = self.preprocessors[s][0].filter_data(self.x[0:2], self.time_window)
        ft = self.preprocessors[s][1].transform(xfb)
        self.dims = [self.nb_segments] + list(ft.shape[1:]) + [1]

    def initialize_preprocessor(self, subject, subject_data, fs):
        logging.info(f"Fitting OVR-FBCSP for subject {subject} ...")
        self.preprocessors[subject] = [
                    FilterBank(fs, self.nb_bands),
                    OVR_CSP(self.NB_CLASSES, self.m_filters)
                ]
        xfb = self.preprocessors[subject][0]\
            .filter_data(subject_data["x_data"], self.time_window)
        self.preprocessors[subject][1].fit(xfb, subject_data["y_labels"])


    def load_external_preprocessors(self, preprocessors: Dict[int, OVR_CSP]):
        self.preprocessors = {}
        for subject, prep in preprocessors.items():
            self.preprocessors[subject] = deepcopy(prep)

    def setup(self):
        logging.info("IV2aDataset setup ...")
        self.build_subject_list()
        self.load_data_and_preprocessors()
        

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
        # NOTE: hardcode
        x = self.x[idx]  # (C, T)
        y = self.y[idx].reshape((-1, ))
        s = self.s[idx]

        if self.transform is not None:
            x = self.transform(x)
        # xfb (B, 1, C, T)
        xfb = self.preprocessors[s][0].filter_data(x[None], self.time_window)
        # === split signals to multiple segments
        # zero padding
        pad_width = self.nb_segments - x.shape[-1] % self.nb_segments
        pad_before = pad_width // 2
        pad_after = pad_width - pad_before
        xfb = np.pad(xfb, ((0, 0), (0, 0), (0, 0), (pad_before, pad_after)), mode="mean")
        # split to sequence
        seglen = xfb.shape[-1] // self.nb_segments
        segments = []
        for i in range(self.nb_segments):
            segments.append(xfb[..., seglen*i:seglen*(i+1)])
        segments = np.concatenate(segments, axis=1) # (B, 4, C, T)

        # preprocess each segment with OVR-FBCSP
        features = self.preprocessors[s][1].transform(segments)  # (nb_segments, B, M)
        features = np.expand_dims(features, 1) # (Nseg, 1, B, M)
        # features *= -1.0

        sample = {
            "eeg": x.astype(np.float32),
            "ovr-fbcsp": features.astype(np.float32),
            "y": y,
        }

        sample = ToTensor()(sample)

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