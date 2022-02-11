"""
The script to run inference on a given GDF file
"""
import argparse
import logging
import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from bci_deep.bcic_iv2a.data_reader import IV2aReader

import bci_deep.model.config as config_collection
import bci_deep.model
from bci_deep.preprocess import FilterBank

logging.getLogger().setLevel(logging.INFO)


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gdf", type=str, required=True,
                        help="Path to the GDF file")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to checkpoint")
    parser.add_argument("--label", type=str, default=None,
                        help="Path to .mat file for label if needed")
    parser.add_argument("--config", type=str, default="hdnn_all_da",
                        help="name of config function in hdnn/config.py. Required if you use a custom config")
    parser.add_argument("--lightning_module", type=str, default="LitModel",
                        help="Name of the Lightning Module subclass. Default: `LitModel`")
    return parser


def main(args):
    # Build experiment config from config name
    logging.info(f"Load config {args.config}")
    config = getattr(config_collection, args.config)()
    # Instantiate Lightning Module from config
    lit_model_class = getattr(bci_deep.model, args.lightning_module)
    lit_model = lit_model_class(**config)
    lit_model = lit_model.load_from_checkpoint(args.ckpt)
    # Read gdf file
    iv2a = IV2aReader()
    data = iv2a.read_file(args.gdf, matfile=args.label, tmin=config.tmin, tmax=config.tmax)
    eeg = data["x_data"] # (N, C, T)
    # Preprocess data
    fb = FilterBank(**config)
    eeg = config.test_transform(eeg)
    eeg_fb = fb.np_forward(eeg) # (N, C, B, T)
    eeg_fb = np.moveaxis(eeg_fb, 2, 1) # (N, B, C, T)
    eeg_fb = torch.tensor(eeg_fb).to(torch.float32)
    # Inference
    ypred = []
    for x in eeg_fb:
        ypred.append(lit_model.predict_step(x[None]))
    ypred = torch.cat(ypred)
    # Metrics if possible
    if data["y_labels"] is not None:
        target = torch.tensor(data["y_labels"])
        lit_model.kappa(ypred, target)
        lit_model.accuracy(ypred, target)
        lit_model.confusion(ypred, target)
        kappa = lit_model.kappa.compute().numpy()
        accuracy = lit_model.accuracy.compute().numpy()
        df_cm = pd.DataFrame(
            lit_model.confusion.compute().numpy(),
            index=range(config.nb_classes),
            columns=range(config.nb_classes))
        logging.info(f"KAPPA: {kappa:.3f}")
        logging.info(f"ACCURACY: {accuracy:.3f}")
        logging.info(f"CONFUSION MATRIX:\n{df_cm}")
    logging.info("Inference Done")



if __name__ == "__main__":
    parser = get_argument_parser()
    args = parser.parse_args()
    main(args)