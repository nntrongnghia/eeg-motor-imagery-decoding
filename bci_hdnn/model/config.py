import torch.nn as nn
from ml_collections import ConfigDict

import bci_hdnn.bcic_iv2a.transform as T
from bci_hdnn.model import HDNN, MLP


def hdnn_base_config():
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
    cfg.train_transform = T.eeg_augmentation(cfg.temporal_size, cfg.noise_srn)
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

    # LightningModule config
    cfg.lr = 0.001

    # Trainer
    cfg.trainer_kwargs = {
        "stochastic_weight_avg": True
    }

    return cfg


def hdnn_base_no_da_config():
    cfg = hdnn_base_config()
    del cfg.train_transform
    del cfg.test_transform
    return cfg
