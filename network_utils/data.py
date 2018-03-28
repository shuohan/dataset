# -*- coding: utf-8 -*-

from .loads import load


class Data3d:
    """Object handling a 3D medical image

    """
    def __init__(self, filepath, load_on_the_fly=True, transpose4d=True):

        self.filepath = filepath
        self.load_on_the_fly = load_on_the_fly
        self.transpose4d = transpose4d

        self._data = None

    def get_data(self):
        """Get the data
        
        Returns:
            data (num_channels x num_i ... numpj.array): The loaded data

        Raises:
            TypeError: The data is not 3D or 4D

        """
        if self.load_on_the_fly:
            return self._load()
        else:
            if self._data is None:
                self._data = self._load()
            return self._data

    def _load(self):
        """Load data

        Returns:
            data (num_channels x num_i ... numpj.array): The loaded data
        
        Raises:
            TypeError: The data is not 3D or 4D

        """
        data = load(self.filepath)
        if len(data.shape) == 4:
            if self._transpose:
                data = np.transpose(data, [3, 0, 1, 2])
        elif len(data.shape == 3):
            data = data[None, ...]
        else:
            raise TypeError('The data should be 3D or 4D (muli-channel 3D).')
        return data
