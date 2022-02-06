import torch.nn as nn
from ml_collections import ConfigDict

import bci_deep.bcic_iv2a.transform as T
from bci_deep.model import HDNN, MLP
from torchvision.transforms import Compose

def hdnn_base():
    cfg = ConfigDict()
    # Data config
    cfg.tmin = 0.0
    cfg.tmax = 4.0
    cfg.nb_classes = 4
    cfg.nb_bands = B = 16 # B
    cfg.nb_segments = L = 4 # L default=4
    cfg.m_filters = 2
    M = 2*cfg.m_filters*cfg.nb_classes

    # Datamodule config
    cfg.num_workers = 3
    cfg.batch_size = 24
    cfg.train_ratio = 1.0

    # Data augmentation
    cfg.temporal_size = int(3.5*250) + 1
    cfg.noise_srn = 40
    cfg.train_transform = Compose([
        T.TemporalRandomCrop(cfg.temporal_size),
        T.GaussianNoise(cfg.noise_srn)
    ])
    cfg.test_transform = T.TemporalCrop(cfg.temporal_size, position="end")

    # HDNN config
    cfg.model_class = HDNN
    cfg.model_kwargs = {
        "input_dims": (L, 1, B, M),
        "nb_segments": cfg.nb_segments,
        "m_filters": cfg.m_filters,
        "cnn1_out_channels": 4,
        "lstm_hidden_size": 32,
        "lstm_output_size": 0, # if 0, output dim = hidden dim
        "lstm_input_size": 32,
        "lstm_num_layers": 3,
        "p_dropout": 0.2,
        "head_hidden_dim": 128,
    }

    # LightningModule optimizer config
    cfg.lr = 0.001


    # Trainer
    cfg.trainer_kwargs = {
        "stochastic_weight_avg": True
    }

    return cfg


def hdnn_base_no_da():
    cfg = hdnn_base()
    del cfg.train_transform
    del cfg.test_transform
    return cfg

def hdnn_norm_no_da():
    cfg = hdnn_base_no_da()
    cfg.train_transform = T.Standardize()
    cfg.test_transform = T.Standardize()
    return cfg

def hdnn_norm_flip():
    cfg = hdnn_base_no_da()
    cfg.train_transform = Compose([
        T.Standardize(),
        T.RandomFlip()
    ])
    cfg.test_transform = T.Standardize()
    return cfg

def hdnn_norm_scale():
    cfg = hdnn_base_no_da()
    cfg.train_transform = Compose([
        T.Standardize(),
        T.RandomScale()
    ])
    cfg.test_transform = T.Standardize()
    return cfg

def hdnn_norm_uninoise():
    cfg = hdnn_base_no_da()
    cfg.train_transform = Compose([
        T.Standardize(),
        T.UniformNoise()
    ])
    cfg.test_transform = T.Standardize()
    return cfg

def hdnn_norm_freqshift():
    cfg = hdnn_base_no_da()
    cfg.train_transform = Compose([
        T.Standardize(),
        T.RandomFrequencyShift()
    ])
    cfg.test_transform = T.Standardize()
    return cfg

def hdnn_norm_gaussnoise():
    cfg = hdnn_base_no_da()
    cfg.train_transform = Compose([
        T.Standardize(),
        T.GaussianNoise(40)
    ])
    cfg.test_transform = T.Standardize()
    return cfg

def hdnn_norm_bar():
    cfg = hdnn_base_no_da()
    cfg.bar_augmentation = True
    cfg.train_transform = T.Standardize()
    cfg.test_transform = T.Standardize()
    return cfg
