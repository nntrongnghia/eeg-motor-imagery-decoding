"""
Experiment configurations.
Feel free to play around
"""
import torch
import torch.nn as nn
import numpy as np
from ml_collections import ConfigDict
import bci_deep.bcic_iv2a.transform as T
from bci_deep.model import HDNN
from torchvision.transforms import Compose
from bci_deep.model.losses import SmoothCECenterLoss
from bci_deep.model.no_filter_hdnn import NoFilterHDNN

def hdnn_no_da():
    cfg = ConfigDict()
    # Datamodule config
    cfg.fs = 250
    cfg.tmin = 0.0
    cfg.tmax = 4.0
    cfg.nb_classes = 4
    cfg.nb_bands = 9 # B
    cfg.nb_segments = 4 # L 
    cfg.m_filters = 1
    cfg.num_workers = 3
    cfg.batch_size = 24

    # HDNN config
    cfg.model_class = HDNN
    cfg.cnn1_out_channels = 4
    cfg.lstm_hidden_size = 32
    cfg.lstm_output_size = 0 # if 0, output dim = hidden dim
    cfg.lstm_input_size = 32
    cfg.lstm_num_layers = 3
    cfg.p_dropout = 0.1
    cfg.head_hidden_dim = 512
    cfg.trainable_csp = False

    # LightningModule config
    cfg.input_key = "eeg_fb"
    cfg.lr = 0.001

    # Trainer
    cfg.trainer_kwargs = {
        "stochastic_weight_avg": True
    }

    return cfg


def no_filter_hdnn_no_da():
    cfg = ConfigDict()
    # Datamodule config
    cfg.fs = 250
    cfg.tmin = 0.0
    cfg.tmax = 4.0
    cfg.nb_classes = 4
    cfg.nb_bands = 0 # 0 means no filter bank
    cfg.nb_segments = 4 # L 
    cfg.m_filters = 1
    cfg.num_workers = 3
    cfg.batch_size = 24

    # HDNN config
    cfg.model_class = NoFilterHDNN
    cfg.nb_eeg_channels = 22
    cfg.time_length = int((cfg.tmax - cfg.tmin)*cfg.fs + 1)
    cfg.cnn1_out_channels = 4
    cfg.lstm_hidden_size = 32
    cfg.lstm_output_size = 0 # if 0, output dim = hidden dim
    cfg.lstm_input_size = 32
    cfg.lstm_num_layers = 3
    cfg.p_dropout = 0.1
    cfg.head_hidden_dim = 512

    # LightningModule config
    cfg.input_key = "eeg"
    cfg.lr = 0.001

    # Trainer
    cfg.trainer_kwargs = {
        "stochastic_weight_avg": True
    }

    return cfg


def hdnn_norm_no_da():
    cfg = hdnn_no_da()
    cfg.train_transform = T.Standardize()
    cfg.test_transform = T.Standardize()
    return cfg


def hdnn_norm_flip():
    cfg = hdnn_no_da()
    cfg.train_transform = Compose([
        T.Standardize(),
        T.RandomFlip()
    ])
    cfg.test_transform = T.Standardize()
    return cfg

def hdnn_norm_scale():
    cfg = hdnn_no_da()
    cfg.train_transform = Compose([
        T.Standardize(),
        T.RandomScale()
    ])
    cfg.test_transform = T.Standardize()
    return cfg


def hdnn_norm_freqshift():
    cfg = hdnn_no_da()
    cfg.train_transform = Compose([
        T.Standardize(),
        T.RandomFrequencyShift()
    ])
    cfg.test_transform = T.Standardize()
    return cfg

def hdnn_norm_uninoise():
    cfg = hdnn_no_da()
    cfg.train_transform = Compose([
        T.UniformNoise(),
        T.Standardize()
    ])
    cfg.test_transform = T.Standardize()
    return cfg

def hdnn_norm_gaussnoise():
    cfg = hdnn_no_da()
    cfg.train_transform = Compose([
        T.GaussianNoise(40),
        T.Standardize()
    ])
    cfg.test_transform = T.Standardize()
    return cfg

def hdnn_norm_bar():
    cfg = hdnn_no_da()
    cfg.bar_augmentation = True
    cfg.train_transform = T.Standardize()
    cfg.test_transform = T.Standardize()
    return cfg


def hdnn_all_da():
    cfg = hdnn_no_da()
    cfg.train_transform = Compose([
        # T.UniformNoise(),
        T.Standardize(),
        T.RandomScale(),
        T.RandomFlip(),
        T.RandomFrequencyShift(),
    ])
    cfg.test_transform = T.Standardize()
    return cfg

def hdnn_no_da_no_dropout():
    cfg = hdnn_no_da()
    cfg.p_dropout = 0.0
    return cfg

def hdnn_all_da_no_dropout():
    cfg = hdnn_all_da()
    cfg.p_dropout = 0.0
    return cfg

def hdnn_random_da():
    cfg = hdnn_no_da()
    cfg.train_transform = Compose([
        T.Standardize(),
        T.RandomChoice([
            T.UniformNoise(),
            T.RandomScale(),
            T.RandomFlip(),
            T.RandomFrequencyShift()
        ])
    ])
    cfg.test_transform = T.Standardize()
    return cfg


def hdnn_random_da_tuned():
    cfg = hdnn_no_da()
    cfg.train_transform = Compose([
        T.Standardize(),
        T.RandomChoice([
            T.UniformNoise(),
            T.RandomScale(),
            T.RandomFlip(),
            T.RandomFrequencyShift()
        ])
    ])
    cfg.test_transform = T.Standardize()
    cfg.p_dropout = 0.1
    cfg.lr = 0.0003
    cfg.cnn1_out_channels = 16
    cfg.batch_size = 4
    return cfg