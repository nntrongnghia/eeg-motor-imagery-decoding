from typing import Tuple
import scipy.linalg
import numpy as np

class CSP:
    def __init__(self,m_filters=2):
        """Common Spatial Pattern spatial filter
        For more details, check doi: 10.3389/fnins.2012.00039

        Parameters
        ----------
        m_filters : int, optional
            Number of filters, by default 2
        """
        self.m_filters = m_filters
        # Transformation matrix, shape (C, 2*m)
        self.Wb = None
        # Full W matrix, shape (C, C)
        self.W = None

    def fit(self, x_data: np.ndarray, y_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Find the transformation matrix Wb in CSP algorithm. 
        This method will instantiate the self.Wb matrix

        Parameters
        ----------
        x_data : np.ndarray
            EEG signals, shape (N, C, T)
            N trials, C channels, T discrete time
        y_labels : np.ndarray
            EEG labels, this should contain only 2 labels [0, 1]
            Shape (N, )

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - eig_values: eigen values, shape (C,)
            - u_mat: eigen vectors in columns, shape (C, C)
        """
        y_labels = y_labels.reshape(-1, )
        classes = np.unique(y_labels)
        assert len(classes) == 2, "y_train should contain only 2 classes"
        n_samples = x_data.shape[-1]

        # center signals
        xc = x_data - x_data.mean(-1, keepdims=True)

        # Find covariance matrixes
        cov_x = []
        for cls in classes:
            x = xc[y_labels == cls]
            xt = np.moveaxis(x, 1, -1)
            cov = (x @ xt) / (n_samples - 1)
            cov_x.append(cov.mean(0))
        
        # Find the transformation matrix W
        cov_x = np.stack(cov_x, axis=0)
        cov_combined = cov_x[0] + cov_x[1]
        eig_values, u_mat = scipy.linalg.eig(cov_combined, cov_x[0])
        sort_indices = np.argsort(np.abs(eig_values))[::-1]
        eig_values = eig_values[sort_indices]
        u_mat = u_mat[:,sort_indices]
        self.W = u_mat
        # Save Wb matrix which is the first m columns and the last m columns of W
        self.Wb = np.concatenate([
            u_mat[:, :self.m_filters],
            u_mat[:, -self.m_filters:]
        ], axis=-1)

        return eig_values, u_mat

    def transform(self, x_trial: np.ndarray) -> np.ndarray:
        """Feature extraction using CSP algorithm

        Parameters
        ----------
        x_trial : np.ndarray
            Single trial of EEG signals, shape (C, T)
            C channels, T discrete time

        Returns
        -------
        np.ndarray
            Feature by CSP, shape (2*m)
            m number of CSP filters
        """
        Z = self.Wb.T @ x_trial
        ZZt = Z @ Z.T
        return np.log(np.diag(ZZt)/np.trace(ZZt))

        