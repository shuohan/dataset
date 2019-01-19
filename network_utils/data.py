# -*- coding: utf-8 -*-

import numpy as np

from .loads import load


class Data:
    """Abstract class to handle data

    Attributes:
        on_the_fly (bool): True if to load data on the fly
        _data (numpy.array): The reference to the loaded data; None if the data
            is not loaded or loaded on the fly

    """
    def __init__(self, on_the_fly=True):
        self.on_the_fly = on_the_fly
        self._data = None

    @property
    def shape(self):
        """Get the shape of the data
        
        Returns:
            shape (tuple of int): The shape of the data

        """
        return self.get_data().shape

    def get_data(self):
        """Get the data

        Reload/reprocess the data if self.on_the_fly, otherwise keep a
        reference to the loaded/processed data.

        Returns:
            data (numpy.array): The loaded/processed etc. data

        """
        if self.on_the_fly:
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

    def cleanup(self):
        """Provide interface for decorator to clean up attributes
        
        Check .data_decorators for more details.

        """
        pass


class Data3d(Data):
    """Object handling a 3D data

    Attributes:
        filepath (str): The path to the file to load
        on_the_fly (bool): True if to load the data on the fly
        transpose4d (bool): Move the last dimension to the first (channel last
            to channel first)

    """
    def __init__(self, filepath, on_the_fly=True, transpose4d=True):
        """Initialize

        """
        super().__init__(on_the_fly)
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
            if self.transpose4d:
                data = np.transpose(data, [3, 0, 1, 2])
        elif len(data.shape) == 3:
            data = data[None, ...]
        else:
            raise TypeError('The data should be 3D or 4D (muli-channel 3D).')
        return data


class DataDecorator(Data):
    """Abstract class to decorate Data

    Attributes:
        data (Data): The decorated data
        on_the_fly (bool): True if to load/process the data on the fly

    """
    def __init__(self, data, on_the_fly=True):
        super().__init__(on_the_fly)
        self.data = data

    @property
    def filepath(self):
        return self.data.filepath

    @property
    def shape(self):
        return self.get_data().shape

    def update(self):
        """Update the state/parameters"""
        self.data.update()

    def cleanup(self):
        """Clean up attributes to save memory"""
        self.data.cleanup()


class Transforming3d(DataDecorator):
    """Transform a data

    Attributes:
        transformer (Transformer): Transform the data

    """
    def __init__(self, data, transformer, on_the_fly=True, **kwargs):
        super().__init__(data, on_the_fly)
        self.transformer = transformer
        self.kwargs = kwargs

    def _get_data(self):
        """Transform the data
        
        Returns:
            transformed (numpy.array): The transformed data

        """
        data = self.transformer.transform(self.data.get_data(), **self.kwargs)
        return data

    def update(self):
        """Update the state/parameters"""
        super().update()
        self.transformer.update()

    def cleanup(self):
        """Cleanup attributes to save memory"""
        super().cleanup()
        self.transformer.cleanup()
