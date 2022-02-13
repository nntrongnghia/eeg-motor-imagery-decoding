from typing import Dict
from ml_collections import ConfigDict
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback

import argparse
import logging
import os
from datetime import datetime

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from bci_deep import DEFAULT_GDF_DATA_DIR

import bci_deep.model.tune_config as tune_config
import bci_deep.model
from bci_deep.bcic_iv2a import IV2aDataModule

# for reproducibility
# pl.seed_everything(42, workers=True)


logging.getLogger().setLevel(logging.INFO)


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("subject", type=str, help="Subject ID to train")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory to BCIC IV 2a dataset")
    parser.add_argument("--num_samples", type=int, default=20, 
                        help="Number of samples in grid search")
    parser.add_argument("--gpu", action="store_true", help="Run on GPU")
    parser.add_argument("--config", type=str, default="hdnn_tune",
                        help="name of config function in hdnn/tune_config.py")
    parser.add_argument("--lightning_module", type=str, default="LitModel",
                        help="Name of the Lightning Module subclass. Default: `LitModel`")
    parser.add_argument("--logdir", type=str, default="tune_results",
                        help="Directory to save Tensorboard logging")
    return parser


def train_lit_model(config:ConfigDict, args, num_epochs=200, num_gpus:int=0):

    data_dir = os.path.expanduser(args.data_dir)
    lit_model_class = getattr(bci_deep.model, args.lightning_module)
    lit_model = lit_model_class(**config)
    single_subject_data = IV2aDataModule(data_dir,
                                        include_subject=[args.subject], **config)
    
    single_subject_data.setup(stage="fit")
    single_subject_data.setup(stage="test")

    lit_model.initialize_csp(single_subject_data)

    logger = TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version=".")

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus,
        logger=logger,
        progress_bar_refresh_rate=0,
        callbacks=[
            TuneReportCallback(["max_val_kappa"],
                on="validation_end"),
            ModelCheckpoint(monitor="val_kappa", mode="max",
                            filename=f"s{args.subject}_finetune_best",
                            dirpath=logger.log_dir)
            ],
        stochastic_weight_avg=True)

    trainer.fit(lit_model,
                single_subject_data.train_dataloader(),
                single_subject_data.test_dataloader())
    trainer.test(lit_model, single_subject_data.test_dataloader())


def tune_mnist_asha(config, args, 
                    num_samples=10, 
                    num_epochs=200, 
                    gpus_per_trial=0):

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=num_epochs//2,
        reduction_factor=2)

    reporter = CLIReporter(
        metric_columns=["max_val_kappa", "training_iteration"])

    train_fn_with_parameters = tune.with_parameters(train_lit_model,
                                                    num_epochs=num_epochs,
                                                    num_gpus=gpus_per_trial,
                                                    args=args)

    resources_per_trial = {"cpu": 2, "gpu": gpus_per_trial}

    analysis = tune.run(train_fn_with_parameters,
        resources_per_trial=resources_per_trial,
        metric="max_val_kappa",
        mode="max",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name=args.config,
        local_dir=args.logdir)

    print("Best hyperparameters found were: ", analysis.best_config)

def main(args):
    # Build experiment config from config name
    config = getattr(tune_config, args.config)()
    # train_lit_model(config.to_dict(), args)
    tune_mnist_asha(config.to_dict(), args, 
                    num_samples=args.num_samples,
                    gpus_per_trial=1 if args.gpu else 0)
    
if __name__ == "__main__":
    parser = get_argument_parser()
    args = parser.parse_args()
    if args.data_dir is None:
        args.data_dir = DEFAULT_GDF_DATA_DIR
    main(args)