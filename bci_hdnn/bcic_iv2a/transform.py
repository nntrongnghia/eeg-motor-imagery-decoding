import random
from typing import Dict, Union
import torch
import torchvision.transforms as T
import numpy as np

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample: Dict[str, np.ndarray]):
        tensors = {
            key: torch.tensor(values)
            for key, values in sample.items()
        }
        return tensors

class TemporalRandomCrop(object):
    """Temporally crop the given frame indices at a random location.
    
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
        T = x.shape[-1]
        if T < self.size:
            raise ValueError("Time length shorter than crop length")
        margin = self.size//2 + 1
        if T - margin <= margin:
            return x
        center_index = random.randint(margin, T - margin)
        start = max(center_index - self.size//2, 0)
        end = start + self.size        
        return x[..., start:end]


class TemporalCrop(object):
    def __init__(self, size, position:Union[str, int]="center"):
        """Temporally crop the given frame 

        Parameters
        ----------
        size: int
            Time size to crop
        position: str or int
            Position to crop
            If str, either: "center", "begin", "end"
            If int: index of the beginning of the slice
        """
        assert position in ["center", "begin", "end"]
        position_enum = {
            "center": 0,
            "begin": 1,
            "end": 2
        }
        self.position = position_enum[position]
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
        if self.position == 0: # center
            center_idx = x.shape[-1] // 2
            start = max(center_idx - self.size//2, 0)
            end = min(start + self.size, x.shape[-1])
        elif self.position == 1: # begin
            start = 0
            end = min(start + self.size, x.shape[-1])
        elif self.position == 2: # end
            end = x.shape[-1]
            start = max(end - self.size, 0)
        else:
            raise ValueError("invalid position")
        return x[..., start:end]


class GaussianNoise(object):
    def __init__(self, snr=20) -> None:
        """Addictive Gaussian Noise with given SNR
        """
        self.snr = snr

    def __call__(self, x: np.ndarray)->np.ndarray:
        xrms = np.abs(x.max() - x.min())/2
        noise_rms = 10**(np.log10(xrms) - self.snr/20)
        noise = np.random.randn(*x.shape)*2*noise_rms
        return x + noise

     
def eeg_augmentation(temporal_size:int, noise_srn=50,**kwargs):
    return T.Compose([
        TemporalRandomCrop(temporal_size),
        GaussianNoise(noise_srn)
    ])
