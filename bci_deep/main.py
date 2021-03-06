"""
The main script to run traininng or test a checkpoint.
"""
import argparse
import logging
import os
from datetime import datetime
import ml_collections

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from bci_deep import DEFAULT_GDF_DATA_DIR
from ml_collections import ConfigDict
import bci_deep.model.config as config_collection
import bci_deep.model
from bci_deep.bcic_iv2a import IV2aDataModule

# for reproducibility
# pl.seed_everything(0, workers=True)


logging.getLogger().setLevel(logging.INFO)


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("subject", type=str, help="Subject ID to train")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory to BCIC IV 2a dataset")
    parser.add_argument("--config", type=str, default="hdnn_all_da",
                        help="name of config function in hdnn/config.py")
    parser.add_argument("--use_transfer_learning", action="store_true",
                        help="If set, no pretrain model with cross subject data")
    parser.add_argument("--test_ckpt", type=str, default=None,
                        help="Path to checkpoint to be tested. If set, no training will be executed.")
    parser.add_argument("--lightning_module", type=str, default="LitModel",
                        help="Name of the Lightning Module subclass. Default: `LitModel`")
    parser.add_argument("--logdir", type=str, default="lightning_logs",
                        help="Directory to save Tensorboard logging")
    parser.add_argument("--disable_early_stopping", action="store_true")
    return parser


def test_model(args, config, lit_model):
    trainer = pl.Trainer.from_argparse_args(args, logger=False)
    single_subject_data = IV2aDataModule(args.data_dir,
                                         include_subject=[args.subject], **config)
    single_subject_data.setup(stage="test")
    trainer.test(lit_model, single_subject_data.test_dataloader())


def train_single_subject(args, config, expe_name, lit_model):
    tb_logger = TensorBoardLogger(args.logdir, name=expe_name,
                                  version=f"finetune_{args.subject}")
    ckpt_name = f"s{args.subject}_finetune_best"
    finetune_ckpt_path = os.path.join(tb_logger.log_dir, ckpt_name+".ckpt")
    callbacks = [
        ModelCheckpoint(monitor="val_kappa", mode="max",
                        filename=ckpt_name,
                        dirpath=tb_logger.log_dir)]
    if not args.disable_early_stopping:
        callbacks += [EarlyStopping(monitor="val_kappa", mode="max", patience=40)]
    single_subject_data = IV2aDataModule(args.data_dir,
                                         include_subject=[args.subject], **config)
    single_subject_data.setup(stage="fit")
    single_subject_data.setup(stage="test")
    lit_model.initialize_csp(single_subject_data)
    trainer = pl.Trainer.from_argparse_args(args,
                                            logger=tb_logger,
                                            callbacks=callbacks,
                                            # val_check_interval=1,
                                            # log_every_n_steps=1,
                                            **config.trainer_kwargs)
    trainer.fit(lit_model,
                single_subject_data.train_dataloader(),
                single_subject_data.test_dataloader())
    lit_model = lit_model.load_from_checkpoint(finetune_ckpt_path)
    # test
    trainer.test(lit_model, single_subject_data.test_dataloader())
    return lit_model


def pretrain_cross_subjects(args, config, expe_name, lit_model):
    tb_logger = TensorBoardLogger(args.logdir, name=expe_name,
                                  version=f"pretrain_{args.subject}")
    ckpt_name = f"s{args.subject}_pretrain_best"
    pretrain_ckpt_path = os.path.join(
        tb_logger.log_dir, ckpt_name+".ckpt")
    callbacks = [
        ModelCheckpoint(monitor="val_loss", mode="min",
                        filename=ckpt_name,
                        dirpath=tb_logger.log_dir)]
    if not args.disable_early_stopping:
        callbacks += [EarlyStopping(monitor="val_loss", mode="min", patience=40)]
    cross_subject_data = IV2aDataModule(args.data_dir,
                                        exclude_subject=[args.subject], **config)
    cross_subject_data.setup(stage="fit")
    cross_subject_data.setup(stage="test")
    lit_model.initialize_csp(cross_subject_data)
    trainer = pl.Trainer.from_argparse_args(args,
                                            logger=tb_logger,
                                            callbacks=callbacks,
                                            # val_check_interval=1,
                                            # log_every_n_steps=1,
                                            **config.trainer_kwargs)
    trainer.fit(lit_model,
                cross_subject_data.train_dataloader(),
                cross_subject_data.test_dataloader())
    lit_model = lit_model.load_from_checkpoint(pretrain_ckpt_path)
    return lit_model


def export_config_to_yaml(args, config, expe_name):
    os.makedirs(os.path.join(args.logdir, expe_name), exist_ok=True)
    config_yaml_path = os.path.join(args.logdir, expe_name, "config.yaml")
    with open(config_yaml_path, "w") as f:
        f.write(config.to_yaml())



def main(args):
    if args.data_dir is None:
        args.data_dir = DEFAULT_GDF_DATA_DIR
    lit_model_class = getattr(bci_deep.model, args.lightning_module)
    # === If train model
    if args.test_ckpt is None:
        # Build experiment config from config name
        logging.info(f"Load config {args.config}")
        config = getattr(config_collection, args.config)()
        logging.info(config)
        # Set the experiment name
        expe_name = "{}_s{}_{:%y-%b-%d-%Hh-%M}".format(
            args.config, args.subject, datetime.now())
        # Instantiate Lightning Module from config
        lit_model = lit_model_class(**config)
        # save experiment config as yaml file
        export_config_to_yaml(args, config, expe_name)
        if args.use_transfer_learning:
            lit_model = pretrain_cross_subjects(
                args, config, expe_name, lit_model)
            lit_model.finetune()
        lit_model = train_single_subject(args, config, expe_name, lit_model)
    # === Else, load checkpoint for testing
    else:
        logging.info(f"Load checkpoint: {args.test_ckpt}")
        if torch.cuda.is_available():
            ckpt = torch.load(args.test_ckpt)
        else:
            ckpt = torch.load(args.test_ckpt, map_location=torch.device('cpu'))
            
        config = ConfigDict()
        config.update(ckpt["hyper_parameters"])
        lit_model = lit_model_class.load_from_checkpoint(args.test_ckpt)
        test_model(args, config, lit_model)


if __name__ == "__main__":
    parser = get_argument_parser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
