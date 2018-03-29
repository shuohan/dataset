# -*- coding: utf-8 -*-

from .loads import load


class Data:

    def __init__(self, get_data_on_the_fly=True):
        self.get_data_on_the_fly = get_data_on_the_fly
        self._data = None

    def get_data(self):
        """Get the data

        Reload/reprocess the data if self.get_data_on_the_fly, otherwise keep a
        reference to the loaded/processed data.

        Returns:
            data (numpy.array): The loaded/processed etc. data

        """
        if self.get_data_on_the_fly:
            return self._get_data()
        else:
            if self._data is None:
                self._data = self._get_data()
            return self._data

    def _get_data(self):
        """Abstract method to load/process the data

        self.get_data() calls this method to get the data.

        Returns:
            data (numpy.array): The loaded/processed etc. data

        """
        raise NotImplementedError

    def update(self):
        """Provide interface for decorator to update parameters
        
        Check .data_decorators for more details.

        """
        pass


class Data3d(Data):
    """Object handling a 3D data

    """
    def __init__(self, filepath, get_data_on_the_fly=True, transpose4d=True):
        super().__init__(get_data_on_the_fly)
        self.filepath = filepath
        self.transpose4d = transpose4d

    def _get_data(self):
        """Load 3D/4D (multi-channel 3D) data from hard drive

        Call external load function to handle different file types when loading
        the data. Assume channels first.

        Returns:
            data (num_channels x num_i ... numpy.array): The loaded data
        
        Raises:
            TypeError: The data is not 3D or 4D

        """
        data = load(self.filepath)
        if len(data.shape) == 4:
            if self._transpose:
                data = np.transpose(data, [3, 0, 1, 2])
        elif len(data.shape) == 3:
            data = data[None, ...]
        else:
            raise TypeError('The data should be 3D or 4D (muli-channel 3D).')
        return data
