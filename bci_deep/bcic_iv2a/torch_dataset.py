"""
A subclass of PyTorch Dataset to load data.
"""
import pickle
import os
import logging
from typing import Dict, List
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from bci_deep import DEFAULT_MAT_DATA_DIR
from bci_deep.bcic_iv2a import IV2aGdfReader
from bci_deep.bcic_iv2a.data_reader import IV2aMatReader
from bci_deep.preprocess import OVR_CSP, FilterBank
from torch.utils.data import Dataset
from bci_deep.bcic_iv2a.transform import ToTensor
import json
import random

class IV2aDataset(Dataset):
    NB_CLASSES = 4
    FS = 250
    NB_SAMPLES_PER_SUBJECT = 288

    def __init__(self,
                 data_dir=None, train: bool = True, nb_bands=9, 
                 f_width=4.0, f_min=4.0, f_max=40.0, f_trans=2.0, 
                 gpass=3.0, gstop=30.0,
                 include_subject: List[str] = [], 
                 exclude_subject: List[str] = [],
                 tmin=0.0, tmax=4.0, transform=None,
                 **kwargs) -> None:
        """Subclass of PyTorch Dataset to load data for training/testing
        This class will export EEG signals and labels to npz files

        Parameters
        ----------
        data_dir : str
            Path to dataset directory
        train : bool, optional
            If True, load training data "T",
            or else, load evaluation data "E"
            By default True
        nb_bands : int, optional
            Number of bands in Filter Bank, by default 9
        f_width : float, optional
            Bandwidth of passband filter in Filter Bank, in Hz
            By default 4
        f_min : float, optional
            Filter Bank lower limit in Hz, by default 4
        f_max : float, optional
            Filter Bank lower limit in Hz, by default 40
        f_trans : float, optional
            Transition between stopband and passband, in Hz
            By default 2.0
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
        transform : [type], optional
            Transformation to apply to EEG raw signals.
            Check bci_deep/bcic_iv2a/transform.py for inspiration
            By default None
        """
        super().__init__()
        if nb_bands > 0:
            self.filter = FilterBank(self.FS, nb_bands, f_width, f_min, f_max, f_trans, gpass, gstop)
        else:
            self.filter = None
        self.tmin, self.tmax = tmin, tmax
        self.train = train
        self.include_subjects = include_subject
        self.exclude_subjects = exclude_subject
        # self.datareader = IV2aGdfReader(data_dir)
        self.datareader = IV2aMatReader(data_dir or DEFAULT_MAT_DATA_DIR)
        self.transform = transform
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
        self.x = []
        self.y = []


    def build_subject_list(self):
        """Get list of subjects of interest
        If `self.include_subjects` is not empty, we take only data of these subjects
        If `self.exclude_subjects` is not empty, data of these subjects will not be included
        """
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


    def load_data(self):
        self.x = []
        self.y = []
        if self.train:
            filenames = [name for name in self.datareader.filenames if "T" in name]
        else:
            filenames = [name for name in self.datareader.filenames if "E" in name]
        for name in filenames:
            subject = int(name[1:3])
            if subject not in self.subject_list:
                continue
            # check if sample files exist
            data = self.datareader.read_file(name, self.tmin, self.tmax)
            self.x.append(data["x_data"])
            self.y.append(data["y_labels"])
        self.x = np.concatenate(self.x)
        self.y = np.concatenate(self.y)

    def setup(self):
        logging.info("IV2aDataset setup ...")
        self.build_subject_list()
        self.load_data()

    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]  # (C, T)
        y = self.y[idx]

        if self.transform is not None:
            x = self.transform(x)
        
        sample = {
            "y": y,
            "eeg": x.astype(np.float32)
        }
        sample = ToTensor()(sample)
        
        if self.filter is not None:
            xfb = self.filter.np_forward(x) # (C, B, T)
            xfb = np.moveaxis(xfb, 1, 0) # (B, C, T)
            xfb = torch.tensor(xfb)
            sample["eeg_fb"] = xfb.to(torch.float32)

        return sample
