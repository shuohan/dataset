# -*- coding: utf-8 -*-

"""Data loaders

Load data from hard drive

"""

import numpy as np


class DataLoader3D:
    """Load data from hard drive. Abstract class

    Attributes:
        filepath (str): The path to the file to load
        load_on_the_fly (bool): If True, reload the data each time `self.data`
            is accessed. Otherwise, only load the data at the first access and
            keep track of it
        data (numpy.array): Access the loaded data
        _load_func (function): The function used to load the file
        _transpose (bool): Transpose the data to channel first
        _data (numpy.array): Loaded data. None means it has not been loaded or
            it is not kept track

    """
    def __init__(self, filepath, load_on_the_fly=False, load_func=load_npy,
                 transpose=True):
        """Initilize

        Args:
            filepath (str): The path to the file to load
        load_on_the_fly (bool): If True, reload the data each time `self.data`
            is accessed. Otherwise, only load the data at the first access and
            keep track of it
        load_func (function): The function used to load the file
        transpose (bool): Assume the loaded data is channel last, therefore
            transposing the data to channel first

        """
        self.filepath = filepath
        self.load_on_the_fly = load_on_the_fly
        self._load_func = load_func
        self._transpose = transpose
        self._data = None

    @property
    def data(self):
        """Access the loaded data
        
        Returns:
            data (numpy.array): Loaded data

        """
        if self.load_on_the_fly:
            return self._load()
        else:
            if self._data is None:
                self._data = self._load()
            return self._data

    def _load(self):
        """Load data from hard drive. Need to override

        Returns:
            data (num_channels x num_i ... numpy.array): The loaded data

        """
        data = self._load_func(self.filepath)
        if len(data.shape) == 4:
            if self._transpose:
                data = np.transpose(data, [3, 0, 1, 2])
        else:
            data = data[None, ...]
        return data


def load_npy(filepath):
    """Load .npy file
    
    Args:
        filepath (str): The path to the file to load

    Returns:
        data (numpy.array): Loaded data

    """
    return np.load(filepath)


def load_nii(filepath):
    """Load .nii/.nii.gz file

    Args:
        filepath (str): The path to the file to load

    Returns:
        data (numpy.array): Loaded data

    """
    import nibabel as nib
    return nib.load(filepath).get_data()
