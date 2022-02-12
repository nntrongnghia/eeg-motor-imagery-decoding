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

def hdnn_lr_bs():
    cfg = C.hdnn_all_da()
    with cfg.ignore_type():
        cfg.lr = tune.loguniform(1e-5, 1e-2)
        cfg.batch_size = tune.choice([4, 8, 16, 24, 32, 64])
    return cfg

def hdnn_tune():
    cfg = C.hdnn_all_da()
    with cfg.ignore_type():
        cfg.lr = tune.loguniform(1e-4, 1e-2)
        cfg.batch_size = tune.choice([4, 16, 24, 64])
        cfg.p_dropout = tune.uniform(0.0, 0.5)
        cfg.cnn1_out_channels = tune.choice([2, 4, 8, 16])
        cfg.nb_bands = tune.randint(9, 17)
        cfg.loss_fn = tune.choice([
            L.ce_loss,
            L.SmoothCECenterLoss()
        ])
    return cfg