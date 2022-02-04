import argparse
import logging
import os
from datetime import datetime

import pytorch_lightning as pl
# for reproducibility
pl.seed_everything(42, workers=True)

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

import bci_hdnn.model.config as config_collection
from bci_hdnn.bcic_iv2a import IV2aDataModule
from bci_hdnn.model import LitModel

logging.getLogger().setLevel(logging.INFO)

def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Directory to BCIC IV 2a dataset")
    parser.add_argument("subject", type=str, help="Subject ID to train")
    parser.add_argument("--config", type=str, default=None, 
        help="name of config function in hdnn/config.py")
    parser.add_argument("--overwrite_sample", action="store_true")
    parser.add_argument("--no_pretrain", action="store_true")
    parser.add_argument("--test_ckpt", type=str, default=None)
    return parser

def main(args):
    if args.config is None:
        args.config = "hdnn_base_config"
    config = getattr(config_collection, args.config)()

    lit_model = LitModel(**config)

    expe_name = "{}_s{}_{:%y-%b-%d-%Hh-%M}".format(args.config, args.subject, datetime.now())

    if args.test_ckpt is None:
        # =====================
        # ==== Pre-train ======
        # =====================
        if not args.no_pretrain:
            tb_logger = TensorBoardLogger("lightning_logs", name=expe_name, 
                version=f"pretrain_{args.subject}")

            ckpt_name = f"s{args.subject}_pretrain_best"
            pretrain_ckpt_path = os.path.join(tb_logger.log_dir, ckpt_name+".ckpt")

            callbacks = [
                EarlyStopping(monitor="val_loss", mode="min", patience=20),
                ModelCheckpoint(monitor="val_kappa", mode="max",
                    filename=ckpt_name,
                    dirpath=tb_logger.log_dir)
            ]

            datamodule_pretrain = IV2aDataModule(args.data_dir, 
                                                exclude_subject=[args.subject], **config, 
                                                overwrite_sample=args.overwrite_sample)    
            datamodule_pretrain.setup(stage="fit")
            datamodule_pretrain.setup(stage="test")
            
            lit_model.initialize_csp(datamodule_pretrain.train_dataloader(), on_gpu=args.gpus is not None)
            
            if args.profiler is not None:
                max_epochs = 2
            else:
                max_epochs = None
            trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger, callbacks=callbacks, max_epochs=max_epochs)
            # trainer.fit(lit_model, datamodule=datamodule_pretrain)



            trainer.fit(lit_model, 
                datamodule_pretrain.train_dataloader(), 
                datamodule_pretrain.test_dataloader())
            del datamodule_pretrain # clean after use

            if args.profiler is not None:
                return

            lit_model.load_from_checkpoint(pretrain_ckpt_path, 
                model_class=config.model_class, 
                model_kwargs=config.model_kwargs)

        # =====================
        # ==== Finetune =======
        # =====================
        # lit_model.finetune()

        tb_logger = TensorBoardLogger("lightning_logs", name=expe_name, 
            version=f"finetune_{args.subject}")

        ckpt_name = f"s{args.subject}_finetune_best"
        finetune_ckpt_path = os.path.join(tb_logger.log_dir, ckpt_name+".ckpt")

        callbacks = [
            EarlyStopping(monitor="val_kappa", mode="max", patience=50),
            ModelCheckpoint(monitor="val_kappa", mode="max",
                filename=ckpt_name,
                dirpath=tb_logger.log_dir,
                verbose=True)
        ]

        datamodule_finetune = IV2aDataModule(args.data_dir, 
                                            include_subject=[args.subject], **config,
                                            overwrite_sample=args.overwrite_sample)
        datamodule_finetune.setup(stage="fit")
        datamodule_finetune.setup(stage="test")

        lit_model.initialize_csp(datamodule_finetune.train_dataloader(), on_gpu=args.gpus is not None)

        trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger, callbacks=callbacks)
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
    lit_model.load_from_checkpoint(finetune_ckpt_path, 
        model_class=config.model_class, 
        model_kwargs=config.model_kwargs)
    datamodule_finetune.setup(stage="test")
    trainer.test(lit_model, datamodule=datamodule_finetune)

if __name__ == "__main__":
    
    parser = get_argument_parser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
