from ml_collections import ConfigDict

from bci_hdnn.bcic_iv2a.transform import eeg_augmentation
from bci_hdnn.hdnn.model import HDNN

def hdnn_base_config():
    cfg = ConfigDict()
    
    # Data config
    cfg.tmin = 0.0
    cfg.tmax = 4.0
    cfg.nb_bands = B = 16 # B
    cfg.nb_segments = L = 4 # L default=4
    cfg.m_filters = 2
    cfg.nb_classes = 4
    M = 2*cfg.m_filters*cfg.nb_classes

    # Datamodule config
    cfg.batch_size = 24
    cfg.train_ratio = 0.8

    # Data augmentation
    cfg.temporal_size = int(3*250)
    cfg.train_transform = eeg_augmentation(cfg.temporal_size)

    # HDNN config
    cfg.input_dims = (L, 1, B, M)
    cfg.cnn1_out_channels = 4
    cfg.lstm_hidden_size = 32
    cfg.lstm_output_size = 0 # if 0, output dim = hidden dim
    cfg.lstm_input_size = 32
    cfg.lstm_num_layers = 3
    cfg.p_dropout = 0.2
    cfg.head_hidden_dim = 512

    # LightningModule config
    cfg.lr = 0.001

    cfg.model = HDNN(**cfg)

    return cfg