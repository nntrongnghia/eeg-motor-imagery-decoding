# TODO: data augmentation for EEG ?
import logging
from typing import Dict, List
from copy import deepcopy
import numpy as np
import torch
from bci_hdnn.bcic_iv2a import BCIC_IV2a
from bci_hdnn.preprocess import OVR_FBCSP
from torch.utils.data import Dataset, random_split

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample: Dict[str, np.ndarray]):
        tensors = {
            key: torch.from_numpy(values)
            for key, values in sample.items()
        }
        return tensors

class IV2aDataset(Dataset):
    NB_CLASSES = 4

    def __init__(self,
                 data_dir, nb_segments=4, train: bool = True,
                 include_subject: List[str] = [], exclude_subject: List[str] = [],
                 tmin=0.0, tmax=4.0, transform=None) -> None:
        super().__init__()
        self.nb_segments = nb_segments
        self.tmin, self.tmax = tmin, tmax
        self.train = train
        self.include_subjects = include_subject
        self.exclude_subjects = exclude_subject
        self.dataset = BCIC_IV2a(data_dir)
        self.x, self.y, self.s = None, None, None
        self.dims = None
        if transform is None:
            self.transform = ToTensor()
        else:
            self.transform = transform
        # self.setup()

    def build_subject_list(self):
        self.subject_list = []
        for filename in self.dataset.filenames:
            s = filename[1:3]
            if len(self.include_subjects) > 0:
                if s not in self.include_subjects:
                    continue
            if len(self.exclude_subjects) > 0:
                if s in self.exclude_subjects:
                    continue
            if s not in self.subject_list:
                self.subject_list.append(int(s))

    def load_data_and_preprocessors(self):
        data = []
        self.preprocessors = {s: None for s in self.subject_list}
        if self.train:
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
            self.preprocessors[subject] = OVR_FBCSP(
                self.NB_CLASSES, subject_data["fs"])
            if self.train:
                logging.info(f"Fitting OVR-FBCSP for subject {subject} ...")
                self.preprocessors[subject].fit(
                    subject_data["x_data"], subject_data["y_labels"])

        self.x = np.concatenate([d["x_data"] for d in data]) # (N, C, T)
        self.y = np.concatenate([d["y_labels"] for d in data]) # (N, )
        self.s = np.concatenate([d["subject"] for d in data]) # (N, )

        # Get input dimensions
        ft = self.preprocessors[1].transform(self.x[0:2])
        self.dims = [self.nb_segments] + list(ft.shape[1:]) + [1]

    def load_external_preprocessors(self, preprocessors: Dict[int, OVR_FBCSP]):
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
        x = self.x[idx]  # (C, T)
        y = self.y[idx].reshape((-1, ))
        s = self.s[idx]

        # === split signals to multiple segments
        # zero padding
        pad_width = self.nb_segments - x.shape[-1] % self.nb_segments
        pad_before = pad_width // 2
        pad_after = pad_width - pad_before
        x = np.pad(x, ((0, 0), (pad_before, pad_after)), mode="mean")
        # split to sequence
        seglen = x.shape[-1] // self.nb_segments
        segments = []
        for i in range(self.nb_segments):
            segments.append(x[:, seglen*i:seglen*(i+1)])
        segments = np.stack(segments)  # (nb_segments, C, T)

        # preprocess each segment with OVR-FBCSP
        features = self.preprocessors[s].transform(segments)  # (nb_segments, B, M)
        features = np.expand_dims(features, 1) # (Nseg, 1, B, M)

        sample = {
            "ft": features,
            "y": y
        }

        sample = self.transform(sample)

        return sample


# test code
if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    dataset = IV2aDataset("/home/nghia/dataset/BCI_IV_2a")
    dataset.setup()
    print(len(dataset))
    print(dataset[0])
    print("Done")
