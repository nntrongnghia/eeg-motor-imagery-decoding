"""
Inspired from https://github.com/fbcsptoolbox/fbcsp_code
"""
import glob
import logging
import os

import mne
import numpy as np
import scipy.io as io


class BCIC_IV2a:
    stimcodes = ('769', '770', '771', '772')
    channels_to_remove = ['EOG-left', 'EOG-central', 'EOG-right']

    def __init__(self, data_dir: str):
        """
        Dataloader for BCI Competition IV 2a Dataset

        The directory structure should be:

        data_dir
        |__true_labels
        |   |__A01T.mat
        |   |__A01E.mat
        |   |__...
        |
        |__A01T.gdf
        |__A01E.gdf
        |__...

        Parameters
        ----------
        data_dir : str
            Directory containing .gdf files
        """
        self.data_dir = data_dir
        self.filenames = [name for name in os.listdir(
            self.data_dir) if name.endswith(".gdf")]

    def load_raw_data_gdf(self, filename: str):
        assert filename in self.filenames
        filepath = os.path.join(self.data_dir, filename)
        logging.info(f"Read {filepath}")
        return mne.io.read_raw_gdf(filepath, verbose="ERROR")

    def read_file(self, filename: str, tmin=0.0, tmax=3.0, baseline=None):
        """Get data as np.ndarray from a given filename

        Parameters
        ----------
        filename : str
            GDF filename
        tmin, tmax: float
            Start and end time of the epochs in seconds, 
            relative to the time-locked event. 
            Defaults to 0.0 and 3.0 respectively (based on BCIC IV 2a description)
            (TODO: check 10.3389/fnins.2012.00039 for time segment choice)
        baseline : [type], optional
            Baseline correction. 
            For more details: https://mne.tools/stable/generated/mne.Epochs.html

        Returns
        -------
        dict
            "x_data": np.ndarray, float
                shape (nb trials, nb channels, time length)
                with time length = `fs`*(`tmax` - `tmin`)
            "y_labels": np.ndarray, int
                shape (nb trials)
                labels for each trial
            "fs": float
                sample frequency
            "subject": str
                subject from whom EEG signals are acquired
        """
        subject = filename[1:3]
        raw_data = self.load_raw_data_gdf(filename)
        sample_freq = raw_data.info.get('sfreq')
        events, event_ids = mne.events_from_annotations(raw_data)

        if "T" in filename:
            stims = [value for key, value in event_ids.items()
                     if key in self.stimcodes]
        elif "E" in filename:
            stims = [value for key, value in event_ids.items() if key == "783"]
        else:
            raise ValueError(
                f"filename {filename} should contain the letter E or T")

        epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=tmin, tmax=tmax,
                            event_repeated='drop', baseline=baseline, preload=True,
                            proj=False, reject_by_annotation=False, verbose="ERROR")
        epochs = epochs.drop_channels(self.channels_to_remove)

        if "T" in filename:
            y_labels = epochs.events[:, -1] - min(epochs.events[:, -1])
        elif "E" in filename:
            mat = io.loadmat(os.path.join(
                self.data_dir, "true_labels", f"A{subject}E.mat"))
            y_labels = mat["classlabel"].reshape(-1) - 1
        else:
            raise ValueError(
                f"filename {filename} should contain the letter E or T")

        x_data = epochs.get_data()*1e6
        eeg_data = {'x_data': x_data,
                    'y_labels': y_labels,
                    'fs': sample_freq,
                    "subject": subject}
        return eeg_data


# test code
if __name__ == "__main__":
    data_dir = "/home/nghia/dataset/BCI_IV_2a"
    bci = BCIC_IV2a(data_dir)
    epochs = []
    files = [name for name in bci.filenames if "E" in name]
    data = bci.read_file(files[3])
    # for filename in files:
    #     data = bci.read_file(filename)
    #     epochs.append(data)
