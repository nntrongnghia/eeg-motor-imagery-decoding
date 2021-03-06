"""
Experiment configurations.
Feel free to play around
"""
import torch
import torch.nn as nn
from ml_collections import ConfigDict
from ray import tune

import bci_deep.bcic_iv2a.transform as T
from bci_deep.model import HDNN
from torchvision.transforms import Compose
import bci_deep.model.config as C
import bci_deep.model.losses as L
from functools import partial

def hdnn_lr_bs():
    cfg = C.hdnn_all_da()
    with cfg.ignore_type():
        cfg.lr = tune.loguniform(1e-5, 1e-2)
        cfg.batch_size = tune.choice([4, 8, 16, 24, 32, 64])
    return cfg

def hdnn_tune():
    cfg = C.hdnn_all_da()
    with cfg.ignore_type():
        cfg.lr = tune.loguniform(1e-5, 1e-3)
        cfg.batch_size = tune.choice([2, 4, 8, 16, 32, 64, 128])
        cfg.cnn1_out_channels = tune.choice([2, 4, 8, 16])
        cfg.nb_bands = tune.randint(9, 17)
        cfg.head_hidden_dim = tune.choice([32, 64, 128, 256, 512])
        # cfg.trainable_csp = tune.choice([True, False])
        cfg.m_filters = tune.randint(1, 10)
    return cfg
