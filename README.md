# BCI - EEG Motor Imagery Decoding using Deep Learning 

This repository would be a great starting point for anyone who want to explore EEG motor imagery decoding using Deep Learning. One can easily play with hyperparameters and implement their own model with minimal effort. Because the data pipeline (dataloader, preprocessing, augmentation) and the training process are already handled.

The well-known BCI Competition IV 2a dataset is used to train and evaluate models. 

Things that are implemented in this repo:
- Common Spatial Pattern algorithm with One-Versus-Rest version for multiclass
- Filter Bank using Chevbychev passband filters
- Dataloader for BCI Competition IV 2a dataset
- Baseline model (CNN + LSTM) based on [this paper](https://doi.org/10.1016/j.bspc.2020.102144)
- Data augmentations for EEG signals
- Grid search for hyperparameter tuning

## Table of contents
1. [Setup](#setup)
2. [Baseline results](#baseline-results)
3. [Usage](#usage)
4. [Want to try your own model? It's easy!](#want-to-try-your-own-model-its-easy)
5. [Contact](#contact)

## Setup

This repo is tested with PyTorch 1.10.2 and Python 3.9.7.

You must add the path to this directory in your PYTHONPATH. If this phrase doesn't sound familiar to you, feel free to check [this tutorial](https://bic-berkeley.github.io/psych-214-fall-2016/using_pythonpath.html#setting-pythonpath-more-permanently).

### Dependencies
You need to install pytorch based on your hardware and environment configuration. Please follow [the official guide](https://pytorch.org/get-started/locally/)

To install other dependencies, run:
```
pip install -r requirements.txt
```

### Dataset
The dataset used in this repo is BCI Competition IV 2a. You can download it [here](https://www.bbci.de/competition/iv/#download). This link doesn't include the evaluation groundtruth, so you need to download it separately [here](https://www.bbci.de/competition/iv/results/index.html#labels).

The dataset directory should be placed in the root of this repo and be structured as follows:
```
./dataset/BCI_IV_2a
|__true_labels
|   |__A01T.mat
|   |__A01E.mat
|   |__...
|
|__A01T.gdf
|__A01E.gdf
|__...
```

## Baseline results

|           | Subject | A01      | A02      | A03      | A04      | A05      | A06      | A07      | A08      | A09      | Avg      |
|-----------|---------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| HDNN      | Acc     | 0.83     | **0.60** | 0.85     | 0.69     | 0.55     | 0.52     | 0.89     | 0.80     | **0.77** | 0.72     |
|           | Kappa   | 0.77     | **0.47** | 0.81     | 0.59     | 0.40     | 0.36     | 0.85     | 0.73     | **0.70** | 0.63     |
|           |         |          |          |          |          |          |          |          |          |          |          |
|           |         |          |          |          |          |          |          |          |          |          |          |
| TunedHDNN | Acc     | **0.83** | 0.58     | **0.90** | **0.70** | **0.64** | **0.56** | **0.90** | **0.82** | 0.76     | **0.74** |
|           | Kappa   | **0.77** | 0.44     | **0.87** | **0.61** | **0.52** | **0.41** | **0.86** | **0.75** | 0.68     | **0.66** |

Checkpoints for the results are saved in `checkpoints` directory of this repo.

- HDNN uses the same hyperparameters in the article [Hybrid deep neural network using transfer learning for EEG motor imagery decoding](https://doi.org/10.1016/j.bspc.2020.102144). Those hyperparameters are resumed in `bci_deep/model/config.py:hdnn_all_da`.
- TunedHDNN's hyperparameters are tuned by grid search. Check `bci_deep/model/config.py:tuned_hdnn_all_da` for details.

## Usage
### Experiment configuration
Each experiment can be configured easily by creating/modifying functions in `bci_deep/model/config.py`. Each configuration contains arguments/hyperparameters the whole pipeline:
- Data module: `IV2aDataModule`
- Data augmentation sequences
- Model's hyperparameters: `LitModel`
- Learning rate
- Trainer's arguments: `pytorch_lightning.Trainer`

For details of each arguments, check the docstrings of corresponding classes.

### Train
```
python bci_deep/main.py SUBJECT [--data_dir DATA_DIR] [--config CONFIG_NAME] [--gpus 1] [--use_transfer_learning]
```

`SUBJECT` is either `01`, `02`, etc. for the subject A01, A02, etc. respectively. 

Options:
- If you place the dataset directory somewhere else than the root of this repo, you should specify it with `--data_dir`
- To run on GPU, add the option `--gpus 1`
- To run training with a specific configuration, add `--config CONFIG_NAME` with `CONFIG_NAME` is the name of a function returning `ml_collection.ConfigDict` defined in `bci_deep/model/config.py`.
- To train model using transfer learning, add `--use_transfer_learning`

Training with transfer learning needs two step: pre-training and fine-tune training. We merged datasets "T" from all the subjects except the subject, who we want to evaluate, to pre-train the model, then we used the datasets "T" of the absent subject in pre-training process to fine-tune. For example, if we want to train the model for the subject 8, we merge datasets "T" from subject 1∼7 and 9 to pre-train, then datasets "T" from subject 8 is used in fine-tune.

The `main.py` runs training with Early Stopping, while training, just grab a coffee or take some air :)

Then the training results (losses, metrics) can be accessed using TensorBoard. The directory `lightning_logs` is supposed to be automatically created in the root path of this project.
```
tensorboard --logdir lightning_logs
```
### Test
To run a test of a checkpoint:
```
python bci_deep/main.py SUBJECT --test_ckpt CHECKPOINT [--config CONFIG_NAME] [--data_dir DATA_DIR]
```
where `CHECKPOINT` is the path to the checkpoint, and `CONFIG_NAME` must be given if you use a custom configuration other than the default one.

### Grid search for hyperparameter tuning
```
python bci_deep/tune.py SUBJECT [--num_samples NUM_SAMPLES] [--config TUNE_CONFIG] [--gpu]
```
Options:
- `--num_samples NUM_SAMPLES` sets the number of random sample in grid search. Default: 20
- `--config TUNE_CONFIG` sets the tuning configuration. Check `bci_deep/model/tune_config.py` for inspiration. Default: `hdnn_tune`
- `--gpu` to run tuning in GPU

The tuning process takes time and it run multiple trial in parallel, so in some case it would be more time efficient if we run the tuning on CPUs.


## Want to try your own model? It's easy!
The data module feeds the trainer with a dictionary of 2 inputs:
- "eeg": (C, T), raw eeg signals
- "eeg_fb": (B, C, T), filtered signals by Filter Bank

With C channels, B filter bands, T time.

To implement your own model, the general guide line should be:
- Implement a model with a similar structure as [`HDNN`](bci_deep/model/hdnn.py)
- Create a configuration function returning `ml_collection.ConfigDict` in `bci_deep/model/config.py`
- That's all!

The rest of the pipeline should works as fine. You could check `no_filter_hdnn.py` and `bci_deep/model/config.py:no_filter_hdnn_no_da` for inspiration.

## Contact
Ngoc Trong Nghia Nguyen - nntrongnghiadt@gmail.com - [Linked In](https://www.linkedin.com/in/ngoc-trong-nghia-nguyen/)
