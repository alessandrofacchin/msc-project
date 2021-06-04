from typing import Tuple
import numpy as np


class DataManager(object):

    @staticmethod
    def split_dataset(data: np.ndarray, train_pct: float=None, val_pct: float=None, test_pct: float=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split Dataset

        Splits a numpy array into train, validation and test based on the first axis.

        Args:
            data (np.ndarray): input dataset
            train_pct (float, optional): percentage of records used for training. Defaults to None.
            val_pct (float, optional): percentage of records used for validation. Defaults to None.
            test_pct (float, optional): percentage of records used for testing. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: training, validation and test datasets
        """
        
        tot_pct = sum([x for x in [train_pct, val_pct, test_pct] if x is not None])
        if tot_pct <= 1:
            missing = sum([x is None for x in [train_pct, val_pct, test_pct]])
            if missing > 1:
                raise ValueError('Cannot handle more than one missing percentage.')
            elif missing == 1:
                [train_pct, val_pct, test_pct] = [x if x is not None else 1 - tot_pct for x in [train_pct, val_pct, test_pct]]
            else:
                train_pct /= tot_pct
                val_pct /= tot_pct
                test_pct /= tot_pct
        else:
            missing = sum([x is None for x in [train_pct, val_pct, test_pct]])
            if missing > 1:
                raise ValueError('Cannot handle more than one missing percentage.')
            elif missing == 1:
                [train_pct, val_pct, test_pct] = [x/100 if x is not None else 1 - tot_pct/100 for x in [train_pct, val_pct, test_pct]]
            else:
                train_pct /= tot_pct
                val_pct /= tot_pct
                test_pct /= tot_pct

        data_copy = data.copy()
        np.random.shuffle(data_copy)
        train, validation, test = np.split(data_copy, 
                       [int(train_pct * data.shape[0]), int((val_pct + train_pct) * data.shape[0])])

        return (train, validation, test)