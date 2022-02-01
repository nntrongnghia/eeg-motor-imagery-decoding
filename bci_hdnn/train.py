import argparse
import os
import pytorch_lightning as pl
import bci_hdnn.hdnn.config as config_collection
from bci_hdnn.hdnn import LitHDNN
from bci_hdnn.bcic_iv2a import IV2aDataModule

def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Directory to BCIC IV 2a dataset")
    parser.add_argument("subject", type=str, help="Subject ID to train")
    parser.add_argument("--config", type=str, default=None, 
        help="name of config function in hdnn/config.py")
    return parser

def main(args):
    if args.config is None:
        args.config = "base_config"
    config = getattr(config_collection, args.config)()
    print(config)
    lit_model = LitHDNN(**config)

    # # === Pre-train
    datamodule = IV2aDataModule(args.data_dir, exclude_subject=[args.subject], **config)
    datamodule.setup(stage="fit")
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(lit_model, datamodule=datamodule)
    # # === Finetune
    # datamodule = IV2aDataModule(args.data_dir, include_subject=[args.subject], **config)
    # datamodule.setup(stage="fit")
    # trainer = pl.Trainer.from_argparse_args(args)
    # trainer.fit(lit_model, datamodule=datamodule)

    # # === Test
    # datamodule.setup(stage="test")
    # trainer.test(lit_model, datamodule=datamodule)

if __name__ == "__main__":
    # for reproducibility
    pl.seed_everything(42, workers=True)
    parser = get_argument_parser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
