import random
from typing import Dict
import torch
import torchvision.transforms as T
import numpy as np

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample: Dict[str, np.ndarray]):
        tensors = {
            key: torch.from_numpy(values)
            for key, values in sample.items()
        }
        return tensors

class TemporalRandomCrop(object):
    """Temporally crop the given frame indices at a random location.
    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.
    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, x: np.ndarray):
        """
        Args:
            x: np.ndarray
                EEG signal, shape (C, T)
        Returns:
            np.ndarray
                EEG signal
        """
        center_index = random.randint(0, x.shape[-1])
        start = max(center_index - self.size, 0)
        end = start + self.size
        return x[..., start:end]
        
def eeg_augmentation(temporal_size:int, **kwargs):
    return T.Compose([
        TemporalRandomCrop(temporal_size),
    ])