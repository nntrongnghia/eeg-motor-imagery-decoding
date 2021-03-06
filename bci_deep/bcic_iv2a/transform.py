"""Transformation to apply on EEG signals
"""
import random
from typing import Dict, List, Union
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
        """Addictive uniform noise
        Let's x the signal
        The noise distribution is [-0.5, 0.5]*x.std()/Cnoise
        """
        self.c = Cnoise
    
    def __call__(self, x:np.ndarray):
        noise = (np.random.rand(*x.shape) - 0.5)*x.std()/self.c
        return x + noise


class Standardize:
    """Normalize signal by mean and std
    """
    def __init__(self, mean=None, std=None, channel_wise=False):
        self.mean = mean
        self.std = std
        self.channel_wise = channel_wise
        
    def __call__(self, x:np.ndarray):
        if (self.mean is None) or (self.std is None):
            if self.channel_wise:
                return (x - x.mean(-1, keepdims=True))/x.std(-1, keepdims=True)
            else:
                return (x - x.mean())/x.std()
        else:
            return (x - self.mean)/self.std


class RandomScale:
    def __init__(self, scale_range=[0.95, 1.05]):
        """Scale signal by a random factor within a given range

        Parameters
        ----------
        scale_range : list, optional
            by default [0.95, 1.05]
        """
        self.scale_range = scale_range
    
    def __call__(self, x:np.ndarray):
        scale = np.random.uniform(*self.scale_range, [*x.shape[:-1], 1])
        return x*scale

class RandomFlip:
    def __init__(self, p=0.5):
        """Randomly flip the signal by a given probability

        Parameters
        ----------
        p : float, optional
            Flip probability, by default 0.5
        """
        self.p = p
    
    def __call__(self, x:np.ndarray):
        if self.p > random.uniform(0, 1):
            return x.max() - x
        else:
            return x

class RandomFrequencyShift:
    def __init__(self, freq_range=[-0.2, 0.2], dt=1/250):
        """Shift the signal frequency by a random offset within a given range

        Parameters
        ----------
        freq_range : list, optional
            by default [-0.2, 0.2]
        dt : float, optional
            Sample time step, by default 1/250 (for BCI IV 2a dataset)
        """
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


class RandomChoice:
    def __init__(self, da_list:List[callable]):
        """Perform random augmentation from the given `da_list`

        Parameters
        ----------
        da_list: List of callable
        """
        self.da_list = da_list

    def __call__(self, x:np.ndarray):
        da_fn = random.choice(self.da_list)
        return da_fn(x)