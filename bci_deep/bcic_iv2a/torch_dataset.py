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
from bci_deep.bcic_iv2a import BCIC_IV2a
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
                 data_dir, train: bool = True, nb_bands=9, 
                 f_width=4.0, f_min=4.0, f_max=40.0, f_trans=2.0, 
                 gpass=3.0, gstop=30.0,
                 include_subject: List[str] = [], 
                 exclude_subject: List[str] = [],
                 tmin=0.0, tmax=4.0, transform=None,
                 sample_dir="sample", overwrite_sample=False,
                 bar_augmentation=False, **kwargs) -> None:
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
        sample_dir : str, optional
            Directory name to save npz files, by default "sample"
            This will be create under `data_dir` 
        overwrite_sample : bool, optional
            If True, rebuild npz files, by default False
        bar_augmentation : bool, optional
            If True, use Brain Area Recombination (BAR) in training.
            For details: https://www.frontiersin.org/articles/10.3389/fnhum.2021.645952/full
            By default False
        """
        super().__init__()
        self.filter = FilterBank(self.FS, nb_bands, f_width, f_min, f_max, f_trans, gpass, gstop)
        self.tmin, self.tmax = tmin, tmax
        self.train = train
        self.include_subjects = include_subject
        self.exclude_subjects = exclude_subject
        self.datareader = BCIC_IV2a(data_dir)
        self.transform = transform
        self.overwrite_sample = overwrite_sample
        self.sample_filenames = None # initialized by self.setup
        self.bar_augmentation = bar_augmentation
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
        """Save dataset information in json file at self.sample_dir
        """
        logging.info(f"Save info.json into {self.sample_dir}")
        json_path = os.path.join(self.sample_dir, "info.json")
        with open(json_path, "w") as f:
            json.dump(self.info, f, indent=4)

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


    def save_sample_files(self):
        """Sample EEG signals and its labels to npz files
        """
        def _save_subject_data(data:dict, subject:int):
            filenames = []
            suffix = "T" if self.train else "E"
            x = data["x_data"] # (N, C, T)
            y = data["y_labels"] # (N, )            
            for i in range(x.shape[0]):
                filename = f"A0{subject}{suffix}_{i+1:03}.npz"
                filenames.append(filename)
                filepath = os.path.join(self.sample_dir, filename)
                logging.info(f"Saving {filepath}")
                np.savez(filepath, x=x[i], y=y[i])
            return filenames
        
        def _build_bar_sample(subject_data: dict, prefix: str):
            def _save_bar_file(left_idx, right_idx):
                left_right_x = np.stack([
                    x[left_idx], x[right_idx]
                ], axis=0) # (2, C, T)
                bar_x = left_right_x[BAR_INDEXING_ARRAY[0], BAR_INDEXING_ARRAY[1]]
                name = f"{prefix}_{left_idx:03}_{right_idx:03}.npz"
                filepath = os.path.join(self.sample_dir, name)
                logging.info(f"Create BAR sample {name}")
                np.savez(filepath, x=bar_x, y=c)
                return name
            # Brain Area Recombination (BAR)
            # First dim: 0 = left, 1 = right
            # Second dim: channel
            BAR_INDEXING_ARRAY = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1], dtype=np.int)
            BAR_INDEXING_ARRAY = np.stack([
                BAR_INDEXING_ARRAY,
                np.arange(0, len(BAR_INDEXING_ARRAY), dtype=BAR_INDEXING_ARRAY.dtype)
            ])
            
            filenames = []
            x = subject_data["x_data"] # (N, C, T)
            y = subject_data["y_labels"] # (N,)
            classes = np.unique(y) 
            for c in classes:
                idx = np.where(y == c)[0]
                for i in range(len(idx)-1):
                    for j in range(i+1, len(idx)):
                        left_idx, right_idx = idx[i], idx[j]
                        name = _save_bar_file(left_idx, right_idx)
                        filenames.append(name)

                        left_idx, right_idx = idx[j], idx[i]
                        name = _save_bar_file(left_idx, right_idx)
                        filenames.append(name)
            
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
            if (len(subject_samples) < self.NB_SAMPLES_PER_SUBJECT) or self.overwrite_sample:
                subject_data = self.datareader.read_file(name, self.tmin, self.tmax)
                self.sample_filenames += _save_subject_data(subject_data, subject)
            else: 
                self.sample_filenames += subject_samples

        # Setup for Brain Area Recombination (BAR) when only in training
        if self.bar_augmentation and self.train:
            assert len(self.subject_list) == 1, "Not support for multiple subjects"
            subject = self.subject_list[0]
            prefix = f"B0{subject}T"
            filenames = os.listdir(self.sample_dir)
            bar_samples = [name for name in filenames if name.startswith(prefix)]
            # check if we need to export files or not
            correct_nb_bar_samples = 4*(self.NB_SAMPLES_PER_SUBJECT/4)*(self.NB_SAMPLES_PER_SUBJECT/4 - 1)
            if (len(bar_samples) < int(correct_nb_bar_samples)) or self.overwrite_sample:
                subject_data = self.datareader.read_file(f"A0{subject}T.gdf", self.tmin, self.tmax)
                bar_samples += _build_bar_sample(subject_data, prefix)
            self.bar_samples = bar_samples
            logging.info("BAR augmentation ready")


    def setup(self):
        logging.info("IV2aDataset setup ...")
        self.build_subject_list()
        self.save_sample_files()

    
    def __len__(self):
        if self.bar_augmentation:
            return len(self.sample_filenames) + self.NB_SAMPLES_PER_SUBJECT
        else:
            return len(self.sample_filenames)

    def __getitem__(self, idx):
        if idx < len(self.sample_filenames):
            sample_filename = self.sample_filenames[idx]
        elif self.bar_augmentation and self.train:
            sample_filename = self.bar_samples[random.randrange(len(self.bar_samples))]
        else:
            raise ValueError("this should not happen")
        
        sample = np.load(os.path.join(self.sample_dir, sample_filename))
        x = sample["x"]  # (C, T)
        y = sample["y"]
        s = int(sample_filename[1:3])
        # double check 
        assert s in self.subject_list

        if self.transform is not None:
            x = self.transform(x)
        
        xfb = self.filter.np_forward(x) # (C, B, T)
        xfb = np.moveaxis(xfb, 1, 0) # (B, C, T)
        xfb = torch.tensor(xfb)

        sample = {
            "y": y,
            "s": s,
            "eeg": x
        }
        sample = ToTensor()(sample)
        sample["eeg_fb"] = xfb.to(torch.float32)
        return sample
