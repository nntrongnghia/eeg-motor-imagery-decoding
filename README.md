# BCI - EEG Motor Imagery Decoding using Deep Learning 

This repository would be a great starting point for anyone who want to explore EEG motor imagery decoding using Deep Learning. One can easily play with hyperparameters and implement their own model with minimal effort. Because the data pipeline (dataloader, preprocessing, augmentation) and the training process are already handled.

The well-known BCI Competition IV 2a dataset is used to train and evaluate models. 

Things that are implemented in this repo:
- Common Spatial Pattern algorithm with One-Versus-Rest version for multiclass
- Filter Bank using Chevbychev passband filters
- Dataloader for BCI Competition IV 2a dataset
- Baseline model (CNN + LSTM) based on [this paper](https://doi.org/10.1016/j.bspc.2020.102144)
- Data augmentations for EEG signals

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

The dataset directory should be structured as follows:
```
DATA_DIR
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

These results are obtained on "E" data using HDNN model with hyperparameters found at `bci_deep/model/config.py:hdnn_all_da`

Checkpoints can be downloaded [HERE](https://drive.google.com/drive/folders/1pJIip2qEzuurFCBFeDy2P4xK8EM1mIOg?usp=sharing)

|         | Subject | A01      | A02      | A03      | A04      | A05      | A06      | A07      | A08      | A09      | Avg      |
|---------|---------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| **HDNN**    | Acc     | **0.88** | 0.61     | 0.87     | **0.76** | **0.65** | **0.58** | **0.90** | 0.85     | **0.81** | **0.77** |
|         | Kappa   | **0.84** | 0.48     | 0.83     | **0.68** | **0.53** | **0.44** | **0.87** | 0.80     | **0.75** | **0.69** |
|         |         |          |          |          |          |          |          |          |          |          |          |
| **HDNN-TL** | Acc     | 0.85     | **0.63** | **0.90** | 0.67     | 0.60     | 0.55     | 0.85     | **0.85** | 0.81     | 0.75     |
|         | Kappa   | 0.80     | **0.50** | **0.87** | 0.56     | 0.47     | 0.40     | 0.80     | **0.81** | 0.75     | 0.66     |

## Usage
### Experiment configuration
Each experiment can be configured easily by creating/modifying functions in `bci_deep/model/config.py`. Each configuration contains arguments/hyperparameters the whole pipeline:
- Data module: `IV2aDataModule`
- Data augmentation sequences
- Model's hyperparameters: `LitModel`
- Learning rate
- Trainer's arguments: `pytorch_lightning.Trainer`

For details of each arguments, check the docstrings of corresponding classes.

### Train and test
Training with transfer learning needs two step: pre-training and fine-tune training. We merged datasets "T" from all the subjects except the subject, who we want to evaluate, to pre-train the model, then we used the datasets "T" of the absent subject in pre-training process to fine-tune. For example, if we want to train the model for the subject 8, we merge datasets "T" from subject 1∼7 and 9 to pre-train, then datasets "T" from subject 8 is used in fine-tune.

```
python bci_deep/main.py DATA_DIR SUBJECT [--config CONFIG_NAME] [--gpus 1] [--no_pretrain]
```

Where `DATA_DIR` is the path to the dataset, `SUBJECT` is either `01`, `02`, etc. for the subject A01, A02, etc. respectively. 

Options:
- To run on GPU, add the option `--gpus 1`
- To run training with a specific configuration, add `--config CONFIG_NAME` with `CONFIG_NAME` is the name of a function returning `ml_collection.ConfigDict` defined in `bci_deep/model/config.py`.
- To skip the pretraining and train the whole model directly on data of the subject we want to evaluate, add `--no_pretrain`

The `main.py` runs training with Early Stopping, while training, just grab a coffee or take some air :)

Then the training results (losses, metrics) can be accessed using TensorBoard. The directory `lightning_logs` is supposed to be automatically created in the root path of this project.
```
tensorboard --logdir lightning_logs
```

To run a test of a checkpoint:
```
python bci_deep/main.py DATA_DIR SUBJECT --test_ckpt CHECKPOINT
```
where `CHECKPOINT` is the path to the checkpoint.

Example in Colab notebook: [here](https://colab.research.google.com/drive/1I2qnpA281TrBaiT9KRdx5_xGsf5_uXZJ?usp=sharing)

## Want to try your own model? It's easy!
The data module feeds the trainer with a dictionary of arrays with key/shape:
- "eeg": (C, T), raw eeg signals
- "eeg_fb": (B, C, T), filtered signals by Filter Bank
- "y": (1,), labels
- "s": (1,), subject id
With C channels, B filter bands, T time.

If your model use filtered signals "eeg_fb", just simply implement your model in PyTorch and create a corresponding configuration. That's all!

If your model use raw eeg signals "eeg", simple additional steps are required:
- Inherit `LitModel`, modify `training/validation/test_step` methods using `batch["eeg"]`
- Make the subclass in the step above visible in `bci_deep/model/__init__.py`
- That's all!

## Contact
Ngoc Trong Nghia Nguyen - nntrongnghiadt@gmail.com - [Linked In](https://www.linkedin.com/in/ngoc-trong-nghia-nguyen/)
