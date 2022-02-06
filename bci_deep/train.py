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
from bci_deep.bcic_iv2a import IV2aDataModule
from bci_deep.model import LitModel

# for reproducibility
pl.seed_everything(42, workers=True)


logging.getLogger().setLevel(logging.INFO)


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str,
                        help="Directory to BCIC IV 2a dataset")
    parser.add_argument("subject", type=str, help="Subject ID to train")
    parser.add_argument("--config", type=str, default=None,
                        help="name of config function in hdnn/config.py")
    parser.add_argument("--overwrite_sample", action="store_true")
    parser.add_argument("--no_pretrain", action="store_true")
    parser.add_argument("--test_ckpt", type=str, default=None)
    return parser


def main(args):
    if args.config is None:
        args.config = "hdnn_base"
    config = getattr(config_collection, args.config)()
    expe_name = "{}_s{}_{:%y-%b-%d-%Hh-%M}".format(
        args.config, args.subject, datetime.now())

    # save experiment config as yaml file
    os.makedirs(os.path.join("lightning_logs", expe_name), exist_ok=True)
    config_yaml_path = os.path.join("lightning_logs", expe_name, "config.yaml")
    with open(config_yaml_path, "w") as f:
        f.write(config.to_yaml())

    lit_model = LitModel(**config)

    if args.test_ckpt is None:
        # =====================
        # ==== Pre-train ======
        # =====================
        if not args.no_pretrain:
            tb_logger = TensorBoardLogger("lightning_logs", name=expe_name,
                                          version=f"pretrain_{args.subject}")

            ckpt_name = f"s{args.subject}_pretrain_best"
            pretrain_ckpt_path = os.path.join(
                tb_logger.log_dir, ckpt_name+".ckpt")

            callbacks = [
                EarlyStopping(monitor="val_loss", mode="min", patience=50),
                ModelCheckpoint(monitor="val_loss", mode="min",
                                filename=ckpt_name,
                                dirpath=tb_logger.log_dir)
            ]

            datamodule_pretrain = IV2aDataModule(args.data_dir,
                                                 exclude_subject=[args.subject], **config,
                                                 overwrite_sample=args.overwrite_sample)
            datamodule_pretrain.setup(stage="fit")
            datamodule_pretrain.setup(stage="test")

            lit_model.initialize_csp(datamodule_pretrain.train_dataloader())

            trainer = pl.Trainer.from_argparse_args(args,
                                                    logger=tb_logger,
                                                    callbacks=callbacks,
                                                    **config.trainer_kwargs)

            trainer.fit(lit_model,
                        datamodule_pretrain.train_dataloader(),
                        datamodule_pretrain.test_dataloader())
            del datamodule_pretrain  # clean after use

            lit_model = lit_model.load_from_checkpoint(pretrain_ckpt_path)
            lit_model.finetune()
        # =====================
        # ==== Finetune =======
        # =====================
        

        tb_logger = TensorBoardLogger("lightning_logs", name=expe_name,
                                      version=f"finetune_{args.subject}")

        ckpt_name = f"s{args.subject}_finetune_best"
        finetune_ckpt_path = os.path.join(tb_logger.log_dir, ckpt_name+".ckpt")

        callbacks = [
            EarlyStopping(monitor="val_kappa", mode="max", patience=100),
            ModelCheckpoint(monitor="val_kappa", mode="max",
                            filename=ckpt_name,
                            dirpath=tb_logger.log_dir),
        ]

        datamodule_finetune = IV2aDataModule(args.data_dir,
                                             include_subject=[args.subject], **config,
                                             overwrite_sample=args.overwrite_sample)
        datamodule_finetune.setup(stage="fit")
        datamodule_finetune.setup(stage="test")

        lit_model.initialize_csp(datamodule_finetune.train_dataloader())

        trainer = pl.Trainer.from_argparse_args(args,
                                                logger=tb_logger,
                                                callbacks=callbacks,
                                                **config.trainer_kwargs)
        # trainer.fit(lit_model, datamodule=datamodule_finetune)
        trainer.fit(lit_model,
                    datamodule_finetune.train_dataloader(),
                    datamodule_finetune.test_dataloader())
    else:
        datamodule_finetune = IV2aDataModule(args.data_dir,
                                             include_subject=[args.subject], **config,
                                             overwrite_sample=args.overwrite_sample)
        trainer = pl.Trainer.from_argparse_args(args)
        finetune_ckpt_path = args.test_ckpt

    # =====================
    # ==== Test ===========
    # =====================
    logging.info(f"Load checkpoint: {finetune_ckpt_path}")
    ckpt = torch.load(finetune_ckpt_path)
    lit_model = lit_model.load_from_checkpoint(finetune_ckpt_path)
    datamodule_finetune.setup(stage="test")
    trainer.test(lit_model, datamodule_finetune.test_dataloader())


if __name__ == "__main__":

    parser = get_argument_parser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
