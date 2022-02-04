import pickle
import os
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
import json

class IV2aDataset(Dataset):
    NB_CLASSES = 4
    FS = 250
    NB_SAMPLES_PER_SUBJECT = 288

    def __init__(self,
                 data_dir, train: bool = True, 
                 nb_bands=9, f_width=4, f_min=4, f_max=40, f_trans=2, gpass=3, gstop=30,
                 include_subject: List[str] = [], 
                 exclude_subject: List[str] = [],
                 tmin=0.0, tmax=4.0, transform=None,
                 sample_dir="sample", overwrite_pkl=False, **kwargs) -> None:
        super().__init__()
        self.filter = FilterBank(self.FS, nb_bands, f_width, f_min, f_max, f_trans, gpass, gstop)
        self.tmin, self.tmax = tmin, tmax
        self.train = train
        self.include_subjects = include_subject
        self.exclude_subjects = exclude_subject
        self.datareader = BCIC_IV2a(data_dir)
        self.transform = transform
        self.overwrite_pkl = overwrite_pkl
        self.sample_filenames = None # initialized by self.setup

        self.sample_dir = os.path.join(data_dir, sample_dir)
        os.makedirs(self.sample_dir, exist_ok=True)

        self.info = {
            "fs": self.FS,
            "nb_bands": nb_bands,
            "f_width": f_width,
            "f_min": f_min,
            "f_max": f_max,
            "f_trans": f_trans,
            "gpass": gpass,
            "gstop": gstop,
            "tmin": tmin,
            "tmax": tmax
        }

    def save_info_as_json(self):
        logging.info(f"Save info.json into {self.sample_dir}")
        json_path = os.path.join(self.sample_dir, "info.json")
        with open(json_path, "w") as f:
            json.dump(self.info, f, indent=4)

    def build_subject_list(self):
        self.subject_list = []
        for filename in self.datareader.filenames:
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


    def save_pickle_files(self):
        def _save_subject_data(data:dict, subject:int):
            filenames = []
            suffix = "T" if self.train else "E"
            x = data["x_data"] # (N, C, T)
            y = data["y_labels"] # (N, )            
            for i in range(x.shape[0]):
                filename = f"A0{subject}{suffix}_{i+1:03}.npz"
                filenames.append(filename)
                filepath = os.path.join(self.sample_dir, filename)
                if not os.path.isfile(filepath) or self.overwrite_pkl:
                    logging.info(f"Saving {filepath}")
                    np.savez(filepath, x=x[i], y=y[i])
            return filenames
        

        self.save_info_as_json()
        self.sample_filenames = []
        if self.train:
            filenames = [name for name in self.datareader.filenames if "T" in name]
            suffix = "T"
        else:
            filenames = [name for name in self.datareader.filenames if "E" in name]
            suffix = "E"
        for name in filenames:
            subject = int(name[1:3])
            if subject not in self.subject_list:
                continue
            # check if sample files exist
            filenames = os.listdir(self.sample_dir)
            prefix = f"A0{subject}{suffix}"
            subject_samples = [name for name in filenames if name.startswith(prefix)]
            if len(subject_samples) < self.NB_SAMPLES_PER_SUBJECT:
                subject_data = self.datareader.read_file(name, self.tmin, self.tmax)
                self.sample_filenames += _save_subject_data(subject_data, subject)
            else: 
                self.sample_filenames += subject_samples


    def setup(self):
        logging.info("IV2aDataset setup ...")
        self.build_subject_list()
        self.save_pickle_files()

    
    def __len__(self):
        return len(self.sample_filenames)

    def __getitem__(self, idx):
        sample_filename = self.sample_filenames[idx]
        sample = np.load(os.path.join(self.sample_dir, sample_filename))
        x = sample["x"]  # (C, T)
        y = sample["y"]
        s = int(sample_filename[1:3])

        if self.transform is not None:
            x = self.transform(x)
        
        # xfb = self.filter(torch.tensor(x)).moveaxis(-2, -3) # (B, C, T)
        xfb = self.filter.np_forward(x) # (C, B, T)
        xfb = np.moveaxis(xfb, 1, 0)
        xfb = torch.tensor(xfb)

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
    print(dataset[0])
    print("done")
    # spliter = StratifiedShuffleSplit(1, train_size=0.8)
    # y = dataset.y + dataset.s*10
    # train_idx, val_idx = next(spliter.split(dataset.x, y))
    # print("Train split", np.unique(y[train_idx], return_counts=True))
    # print("Val split", np.unique(y[val_idx], return_counts=True))
    # print("Done")
