from ml_collections import ConfigDict

def base_config():
    cfg = ConfigDict()
    
    # Data config
    cfg.tmin = 0.0
    cfg.tmax = 4.0
    cfg.nb_bands = B = 9 # B
    cfg.nb_segments = L = 4 # L
    cfg.m_filters = 2
    cfg.nb_classes = 4
    M = 2*cfg.m_filters*cfg.nb_classes

    # Datamodule config
    cfg.batch_size = 8
    cfg.train_ratio = 0.8

    # HDNN config
    cfg.input_dims = (L, 1, B, M)
    cfg.cnn1_out_channels = 4
    cfg.lstm_hidden_size = 32
    cfg.lstm_output_size = 0 # if 0, output dim = hidden dim
    cfg.lstm_input_size = 32
    cfg.lstm_num_layers = 3
    cfg.p_dropout = 0.2

    # LightningModule config
    cfg.lr = 0.001

    return cfg
