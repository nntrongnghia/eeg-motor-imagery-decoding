"""
A class to read EEG signals from .gdf and .mat files
of the BCIC IV 2a dataset.
https://www.bbci.de/competition/iv/#dataset2a
"""
import logging
import os

import mne
import numpy as np
from numpy import squeeze
import scipy.io as io
from bci_deep import DEFAULT_MAT_DATA_DIR


def get_data(subject, training,path):
	'''	Loads the dataset 2a of the BCI Competition IV
	available on http://bnci-horizon-2020.eu/database/data-sets
	Keyword arguments:
	subject -- number of subject in [1, .. ,9]
	training -- if True, load training data
				if False, load testing data
	
	Return:	data_return 	numpy matrix 	size = NO_valid_trial x 22 x 1750
			class_return 	numpy matrix 	size = NO_valid_trial
	'''
	NO_channels = 22
	NO_tests = 6*48 	
	Window_Length = 7*250 

	class_return = np.zeros(NO_tests, dtype=np.int64)
	data_return = np.zeros((NO_tests,NO_channels,Window_Length))

	NO_valid_trial = 0
	if training:
		a = io.loadmat(path+'A0'+str(subject)+'T.mat')
	else:
		a = io.loadmat(path+'A0'+str(subject)+'E.mat')
	a_data = a['data']
	for ii in range(0,a_data.size):
		a_data1 = a_data[0,ii]
		a_data2= [a_data1[0,0]]
		a_data3= a_data2[0]
		a_X 		= a_data3[0] 
		a_trial 	= a_data3[1]
		a_y 		= a_data3[2]
		a_fs 		= a_data3[3]
		a_classes 	= a_data3[4]
		a_artifacts = a_data3[5]
		a_gender 	= a_data3[6]
		a_age 		= a_data3[7]

		for trial in range(0,a_trial.size):
			if(a_artifacts[trial]==0):
				data_return[NO_valid_trial,:,:] = np.transpose(a_X[int(a_trial[trial]):(int(a_trial[trial])+Window_Length),:22])
				class_return[NO_valid_trial] = int(a_y[trial])
				NO_valid_trial +=1

	return data_return[0:NO_valid_trial,:,:], class_return[0:NO_valid_trial]

class IV2aGdfReader:
    stimcodes = ('769', '770', '771', '772')
    channels_to_remove = ['EOG-left', 'EOG-central', 'EOG-right']

    def __init__(self, data_dir: str=None):
        """
        Raw GDF reader for BCI Competition IV 2a Dataset

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
        if data_dir is not None:
            self.filenames = [name for name in os.listdir(
                self.data_dir) if name.endswith(".gdf")]
        else:
            self.filenames = []

    def load_raw_data_gdf(self, filename: str):
        if self.data_dir is not None:
            filepath = os.path.join(self.data_dir, filename)
        else:
            filepath = filename
        logging.info(f"Read {filepath}")
        return mne.io.read_raw_gdf(filepath, verbose="ERROR")

    def read_file(self, filename: str, tmin=0.0, tmax=4.0, baseline=None, matfile=None):
        """Get data as np.ndarray from a given filename

        Parameters
        ----------
        filename : str
            GDF filename
        tmin, tmax: float
            Start and end time in seconds, relative to the start of each cue
            Defaults to 0.0 and 4.0 respectively (based on BCIC IV 2a description)
        baseline : 
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
        gdf_name = os.path.basename(filename).split(".")[0]
        subject = gdf_name[1:3]
        raw_data = self.load_raw_data_gdf(filename)
        sample_freq = raw_data.info.get('sfreq')
        events, event_ids = mne.events_from_annotations(raw_data)

        if "T" == gdf_name[-1]:
            stims = [value for key, value in event_ids.items()
                     if key in self.stimcodes]
        elif "E" == gdf_name[-1]:
            stims = [value for key, value in event_ids.items() if key == "783"]
        else:
            all_stimcodes = self.stimcodes + ["783"]
            stims = [value for key, value in event_ids.items() if key in all_stimcodes]

        epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=tmin, tmax=tmax,
                            baseline=baseline, preload=True, proj=False)
        epochs = epochs.drop_channels(self.channels_to_remove)

        if matfile is not None:
            mat = io.loadmat(matfile)
            y_labels = mat["classlabel"].reshape(-1) - 1
        elif self.data_dir is not None:
            mat = io.loadmat(os.path.join(
                self.data_dir, "true_labels", f"{gdf_name}.mat"))
            y_labels = mat["classlabel"].reshape(-1) - 1
        else:
            y_labels = None

        x_data = epochs.get_data()*1e6
        eeg_data = {'x_data': x_data,
                    'y_labels': y_labels,
                    'fs': sample_freq,
                    "subject": subject}
        return eeg_data


class IV2aMatReader:
    FS = 250
    def __init__(self, data_dir=None):
        """
        Raw .mat data reader for BCI Competition IV 2a Dataset
        available at http://bnci-horizon-2020.eu/database/data-sets

        Parameters
        ----------
        data_dir: str
            Directory containing .mat data downloaded from the url above
        """
        self.data_dir = data_dir
        if self.data_dir is not None:
            self.filenames = [name for name in os.listdir(
                    self.data_dir) if name.endswith(".mat")]
        else:
            self.filenames = []
    
    def read_file(self, filename: str, tmin=0.0, tmax=4.0):
        """Get data as np.ndarray from a given filename

        Parameters
        ----------
        filename : str
            MAT filename
        tmin, tmax: float
            Start and end time in seconds, relative to the start of each cue
            Defaults to 0.0 and 4.0 respectively (based on BCIC IV 2a description)

        Returns
        -------
        dict
            "x_data": np.ndarray, float
                shape (nb trials, nb channels, time length)
                with time length = `fs`*(`tmax` - `tmin`) + 1
            "y_labels": np.ndarray, int
                shape (nb trials)
                labels for each trial
        """
        t1 = int(self.FS*(tmin+2))
        t2 = int(self.FS*(tmax+2))
        if self.data_dir is not None:
            filepath = os.path.join(self.data_dir, filename)
        else:
            filepath = filename
        mat = io.loadmat(filepath, simplify_cells=True)["data"]
        x = []
        y = []
        for data in mat:
            a_X 		= data["X"]
            a_trial 	= data["trial"]
            a_y 		= data["y"]
            for i, trial_idx in enumerate(a_trial):
                x.append(a_X[trial_idx+t1:trial_idx+t2+1,:22].T)
                y.append(a_y[i])
        x = np.stack(x)
        y = np.array(y) - 1
        assert len(y) == len(x)
        return {
            "x_data": x,
            "y_labels": y
        }