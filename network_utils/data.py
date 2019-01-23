# -*- coding: utf-8 -*-
"""Object holding the data loading/processing logic

The attribute Data.value_pairs is awkward since normally an image should not have
a left/right label correspondency but without this the whole stuff is really
messy since the usage of Decorator. The decorator itself is however hard to get
rid of since the transformation requirs the flexibility of the types and the
applying order...

"""
import numpy as np
import warnings

from .loads import load


class Data:
    """Abstract class to handle data

    Attributes:
        on_the_fly (bool): True if to load data on the fly
        interp_order (int): The interpolation order; images should have >=1 and
            label images should have order == 0
        value_pairs (list of tuple): Each element of the list is a two element
            tuple containing the values that need to be swapped during flipping
        _data (numpy.array): The reference to the loaded data; None if the data
            is not loaded or loaded on the fly

    """
    def __init__(self, on_the_fly=True, interp_order=1, value_pairs=list()):
        self.on_the_fly = on_the_fly
        self._interp_order = interp_order
        self._value_pairs = value_pairs
        self._data = None

    @property
    def interp_order(self):
        return self._interp_order

    @interp_order.setter
    def interp_order(self, order):
        self._interp_order = order

    @property
    def value_pairs(self):
        return self._value_pairs

    @value_pairs.setter
    def value_pairs(self, pairs):
        self._value_pairs = pairs

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
    def __init__(self, filepath, on_the_fly=True, transpose4d=True,
                 interp_order=1, value_pairs=list()):
        """Initialize

        """
        super().__init__(on_the_fly, interp_order, value_pairs)
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


class Image3d(Data3d):
    """Object handling a 3D image"""
    def __init__(self, filepath, on_the_fly=True, transpose4d=True):
        super().__init__(filepath, on_the_fly, transpose4d, interp_order=1)


class Label3d(Data3d):
    """Object handling a 3D label image"""
    def __init__(self, filepath, on_the_fly=True, transpose4d=True):
        super().__init__(filepath, on_the_fly, transpose4d, interp_order=0)


class DataDecorator(Data):
    """Abstract class to decorate Data

    Attributes:
        data (Data): The decorated data
        on_the_fly (bool): True if to load/process the data on the fly

    """
    def __init__(self, data, on_the_fly=True):
        super().__init__(on_the_fly)
        self.data = data
        if self.data.on_the_fly and not self.on_the_fly:
            name = self.__class__.__name__
            message = ('Must process on the fly since the wrapped is '
                       'on the fly. Set %s.on_the_fly to True' % name)
            warnings.warn(message, RuntimeWarning),
            self.on_the_fly = True

    @property
    def filepath(self):
        return self.data.filepath

    @property
    def shape(self):
        return self.data.shape # for operation that do not change the shape

    @property
    def interp_order(self):
        return self.data.interp_order

    @property
    def value_pairs(self):
        return self.data.value_pairs

    def update(self):
        """Update the state/parameters"""
        self.data.update()

    def cleanup(self):
        """Clean up attributes to save memory"""
        self.data.cleanup()


class Transforming3d(DataDecorator):
    """Transform the data

    Attributes:
        transformer (transformers.Transformer): Transform the data

    """
    def __init__(self, data, transformer, on_the_fly=True):
        super().__init__(data, on_the_fly)
        self.transformer = transformer

    def _get_data(self):
        """Transform the data
        
        Returns:
            data (numpy.array): The transformed data

        """
        data = self.transformer.transform(self.data.get_data())
        return data

    def update(self):
        """Update the state/parameters"""
        super().update()
        self.transformer.update()

    def cleanup(self):
        """Cleanup attributes to save memory"""
        super().cleanup()
        self.transformer.cleanup()


class Interpolating3d(Transforming3d):
    """Interpolate the data"""
    def _get_data(self):
        dd = self.transformer.transform(self.data.get_data(), self.interp_order)
        return dd


class Flipping3d(Transforming3d):
    """Flip the data"""
    def _get_data(self):
        dd = self.transformer.transform(self.data.get_data(), self.value_pairs)
        return dd


class Cropping3d(Transforming3d):
    """Crop the data"""
    @property
    def shape(self):
        num_channels = self.data.shape[0]
        shape = (num_channels, *self.transformer.cropping_shape)
        return shape
    def _get_data(self):
        return self.transformer.transform(self.data.get_data())
