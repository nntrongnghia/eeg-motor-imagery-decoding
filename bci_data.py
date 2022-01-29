"""
Inspired from https://github.com/fbcsptoolbox/fbcsp_code
"""
import mne
import os
import glob
import numpy as np

class BCIC_IV2a:
    stimcodes = ('769','770','771','772')
    channels_to_remove = ['EOG-left', 'EOG-central', 'EOG-right']

    def __init__(self, data_dir:str):
        """
        Dataloader for BCI Competition IV 2a Dataset

        Parameters
        ----------
        data_dir : str
            Directory containing .gdf files
        """
        self.data_dir = data_dir
        self.filenames = [name for name in os.listdir(self.data_dir) if name.endswith(".gdf")]
        self.train_files = [name for name in self.filenames if "T" in name]
        self.eval_files = [name for name in self.filenames if "E" in name]


    def load_raw_data_gdf(self, filename:str):
        assert filename in self.filenames
        return mne.io.read_raw_gdf(os.path.join(self.data_dir, filename))


    def read_file(self, filename:str, tmin=0.0, tmax=3.0, baseline=None):
        """Get data as np.ndarray from a given filename

        Parameters
        ----------
        filename : str
            GDF filename
        tmin, tmax: float
            Start and end time of the epochs in seconds, 
            relative to the time-locked event. 
            Defaults to 0.0 and 3.0 respectively (based on BCIC IV 2a description)
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
        """
        raw_data = self.load_raw_data_gdf(filename)
        sample_freq = raw_data.info.get('sfreq')
        events, event_ids = mne.events_from_annotations(raw_data)
        stims =[value for key, value in event_ids.items() if key in self.stimcodes]
        epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=tmin, tmax=tmax, event_repeated='drop',
                            baseline=baseline, preload=True, proj=False, reject_by_annotation=False)
        epochs = epochs.drop_channels(self.channels_to_remove)
        y_labels = epochs.events[:, -1] - min(epochs.events[:, -1])
        x_data = epochs.get_data()*1e6
        eeg_data={'x_data':x_data,
                  'y_labels':y_labels,
                  'fs':sample_freq}
        return eeg_data

    def __getitem__(self, idx):
        return self.read_file(self.filenames[idx])


# test code
if __name__ == "__main__":
    data_dir = "D:\Work\dataset\BCI_IV_2a"
    bcic_data = BCIC_IV2a(data_dir)
    eeg = bcic_data.read_file("A01T.gdf")
    print(eeg)