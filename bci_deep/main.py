"""
The main script to run traininng or test a checkpoint.
"""
import argparse
import logging
import os
from datetime import datetime

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

import bci_deep.model.config as config_collection
import bci_deep.model
from bci_deep.bcic_iv2a import IV2aDataModule

# for reproducibility
# pl.seed_everything(0, workers=True)


logging.getLogger().setLevel(logging.INFO)


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str,
                        help="Directory to BCIC IV 2a dataset")
    parser.add_argument("subject", type=str, help="Subject ID to train")
    parser.add_argument("--config", type=str, default="hdnn_all_da",
                        help="name of config function in hdnn/config.py")
    parser.add_argument("--overwrite_sample", action="store_true",
                        help="If set, rebuild IV 2a dataset in npz")
    parser.add_argument("--no_pretrain", action="store_true",
                        help="If set, no pretrain model with cross subject data")
    parser.add_argument("--test_ckpt", type=str, default=None,
                        help="Path to checkpoint to be tested. If set, no training will be executed.")
    parser.add_argument("--lightning_module", type=str, default="LitModel",
                        help="Name of the Lightning Module subclass. Default: `LitModel`")
    return parser


def test_model(args, config, lit_model):
    trainer = pl.Trainer.from_argparse_args(args, logger=False)
    single_subject_data = IV2aDataModule(args.data_dir,
                                         include_subject=[args.subject], **config,
                                         overwrite_sample=args.overwrite_sample)
    single_subject_data.setup(stage="test")
    trainer.test(lit_model, single_subject_data.test_dataloader())


def train_single_subject(args, config, expe_name, lit_model):
    tb_logger = TensorBoardLogger("lightning_logs", name=expe_name,
                                  version=f"finetune_{args.subject}")
    ckpt_name = f"s{args.subject}_finetune_best"
    finetune_ckpt_path = os.path.join(tb_logger.log_dir, ckpt_name+".ckpt")
    callbacks = [
        EarlyStopping(monitor="val_kappa", mode="max", patience=50),
        ModelCheckpoint(monitor="val_kappa", mode="max",
                        filename=ckpt_name,
                        dirpath=tb_logger.log_dir)]
    single_subject_data = IV2aDataModule(args.data_dir,
                                         include_subject=[args.subject], **config,
                                         overwrite_sample=args.overwrite_sample)
    single_subject_data.setup(stage="fit")
    single_subject_data.setup(stage="test")
    lit_model.initialize_csp(single_subject_data)
    trainer = pl.Trainer.from_argparse_args(args,
                                            logger=tb_logger,
                                            callbacks=callbacks,
                                            **config.trainer_kwargs)
    trainer.fit(lit_model,
                single_subject_data.train_dataloader(),
                single_subject_data.test_dataloader())
    lit_model = lit_model.load_from_checkpoint(finetune_ckpt_path)
    # test
    trainer.test(lit_model, single_subject_data.test_dataloader())
    return lit_model


def pretrain_cross_subjects(args, config, expe_name, lit_model):
    tb_logger = TensorBoardLogger("lightning_logs", name=expe_name,
                                  version=f"pretrain_{args.subject}")
    ckpt_name = f"s{args.subject}_pretrain_best"
    pretrain_ckpt_path = os.path.join(
        tb_logger.log_dir, ckpt_name+".ckpt")
    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", patience=50),
        ModelCheckpoint(monitor="val_loss", mode="min",
                                filename=ckpt_name,
                                dirpath=tb_logger.log_dir)]
    cross_subject_data = IV2aDataModule(args.data_dir,
                                        exclude_subject=[args.subject], **config,
                                        overwrite_sample=args.overwrite_sample)
    cross_subject_data.setup(stage="fit")
    cross_subject_data.setup(stage="test")
    lit_model.initialize_csp(cross_subject_data)
    trainer = pl.Trainer.from_argparse_args(args,
                                            logger=tb_logger,
                                            callbacks=callbacks,
                                            **config.trainer_kwargs)
    trainer.fit(lit_model,
                cross_subject_data.train_dataloader(),
                cross_subject_data.test_dataloader())
    lit_model = lit_model.load_from_checkpoint(pretrain_ckpt_path)
    return lit_model


def export_config_to_yaml(config, expe_name):
    os.makedirs(os.path.join("lightning_logs", expe_name), exist_ok=True)
    config_yaml_path = os.path.join("lightning_logs", expe_name, "config.yaml")
    with open(config_yaml_path, "w") as f:
        f.write(config.to_yaml())


def main(args):
    # Build experiment config from config name
    logging.info(f"Load config {args.config}")
    config = getattr(config_collection, args.config)()
    # Set the experiment name
    expe_name = "{}_s{}_{:%y-%b-%d-%Hh-%M}".format(
        args.config, args.subject, datetime.now())
    # Instantiate Lightning Module from config
    lit_model_class = getattr(bci_deep.model, args.lightning_module)
    lit_model = lit_model_class(**config)
    # === If train model
    if args.test_ckpt is None:
        # save experiment config as yaml file
        export_config_to_yaml(config, expe_name)
        if not args.no_pretrain:
            lit_model = pretrain_cross_subjects(
                args, config, expe_name, lit_model)
            lit_model.finetune()
        lit_model = train_single_subject(args, config, expe_name, lit_model)
    # === Else, load checkpoint for testing
    else:
        logging.info(f"Load checkpoint: {args.test_ckpt}")
        lit_model = lit_model.load_from_checkpoint(args.test_ckpt)
        test_model(args, config, lit_model)


if __name__ == "__main__":
    parser = get_argument_parser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
