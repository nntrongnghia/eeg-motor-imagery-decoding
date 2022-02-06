import random
from typing import Dict, Union
import torch
import torchvision.transforms as T
import numpy as np
from scipy.signal import hilbert

class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample: Dict[str, np.ndarray]):
        tensors = {
            key: torch.tensor(values)
            for key, values in sample.items()
        }
        return tensors

class TemporalRandomCrop:
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


class TemporalCrop:
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


class GaussianNoise:
    def __init__(self, snr=20) -> None:
        """Addictive Gaussian Noise with given SNR
        """
        self.snr = snr

    def __call__(self, x: np.ndarray)->np.ndarray:
        xrms = np.abs(x.max() - x.min())/2
        noise_rms = 10**(np.log10(xrms) - self.snr/20)
        noise = np.random.randn(*x.shape)*2*noise_rms
        return x + noise


class UniformNoise:
    def __init__(self, Cnoise=4) -> None:
        self.c = Cnoise
    
    def __call__(self, x:np.ndarray):
        noise = (np.random.rand(*x.shape) - 0.5)*x.std()/self.c
        return x + noise


class Standardize:
    def __call__(self, x:np.ndarray):
        return (x - x.mean())/x.std()


class RandomScale:
    def __init__(self, scale_range=[0.95, 1.05]):
        self.scale_range = scale_range
    
    def __call__(self, x:np.ndarray):
        scale = random.uniform(*self.scale_range)
        return x*scale

class RandomFlip:
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, x:np.ndarray):
        if self.p > random.uniform(0, 1):
            return x.max() - x
        else:
            return x

class RandomFrequencyShift:
    def __init__(self, freq_range=[-0.2, 0.2], dt=1/250):
        self.freq_range = freq_range
        self.dt = dt

    def __call__(self, x:np.ndarray):
        df = random.uniform(*self.freq_range)
        non_temporal_ndim = len(x.shape) - 1
        pad_len = 2 - x.shape[-1] % 2
        pad_width = [(0, 0)]*non_temporal_ndim + [(0, pad_len)]
        padded_x = np.pad(x, pad_width)
        t = np.arange(0, padded_x.shape[-1])*self.dt
        t = t.reshape(*([1]*non_temporal_ndim), -1)
        freq_shift = np.exp(2j*np.pi*df*t)
        shifted_x = hilbert(padded_x)*freq_shift
        shifted_x = shifted_x.real
        return shifted_x[..., :x.shape[-1]]