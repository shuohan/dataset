# -*- coding: utf-8 -*-

import numpy as np
from image_processing_3d import crop3d, calc_bbox3d, resize_bbox3d

from .data import Data


class DataDecorator(Data):
    """Abstract class to decorate Data

    Attributes:
        data (Data): The decorated data

    """
    def __init__(self, data, get_data_on_the_fly=True):
        super().__init__(get_data_on_the_fly)
        self.data = data

    @property
    def filepath(self):
        return self.data.filepath

    def update(self):
        """Update the state/parameters"""
        self.data.update()

    def cleanup(self):
        """Clean up attributes to save memory"""
        self.data.cleanup()


class Cropping3d(DataDecorator):
    """Crop data using mask

    Call externel `image_processing_3d.crop3d` to crop the data. Check its doc
    for more details.

    Attributes:
        data (Data): The data to crop
        mask (Data): The mask used to crop the data
        cropping_shape (tuple of int): The shape of cropped data
        _data (Data): `None` when `self.get_data_on_the_fly` is `True`;
            otherwise holding the cropped data
        _source_bbox (list of slice): The index slices in `self.data` of the
            cropping region
        _target_bbox (list of slice): The index slices in `self._data` (cropped
            `self.data`) of the cropping region

    """
    def __init__(self, data, mask, cropping_shape, get_data_on_the_fly=True):
        super().__init__(data, get_data_on_the_fly)
        self.mask = mask
        self.cropping_shape = cropping_shape

        self._source_bbox = None
        self._target_bbox = None

    def _get_data(self):
        """Crop the data using the corresponding mask

        Returns:
            cropped (num_channels x num_i x ... numpy.array): The cropped data

        """
        data = self.data.get_data()
        mask = self.mask.get_data()[0, ...]
        bbox = calc_bbox3d(mask)
        bbox = resize_bbox3d(bbox, self.cropping_shape)
        cropped, self._source_bbox, self._target_bbox = crop3d(data, bbox)
        return cropped


class Binarizing3d(DataDecorator):
    """Binarize a label image

    Use .label_image_binarizer.LabelImageBinarizer for one-hot encoding.
    
    Attributes:
        binarizer (LabelImageBinarizer): Perform one-hot encoding

    """
    def __init__(self, data, binarizer, get_data_on_the_fly=True):
        super().__init__(data, get_data_on_the_fly)
        self.binarizer = binarizer

    def _get_data(self):
        """Binarize the label image

        Fit the binarizer if not fitted before, and transform the label image.
        Since binarizer returns result with channels last, move the channels
        first. Only support single channel image (1 x num_i x num_j x num_k).

        Returns:
            binarized (num_channels x num_i ... numpy.array): The binarized
                label image

        """
        data = self.data.get_data()[0, ...]
        if not hasattr(self.binarizer, 'classes_'):
            self.binarizer.fit(np.unique(data))
        binarized = self.binarizer.transform(data)
        binarized = np.rollaxis(binarized, -1) # channels first
        return binarized


class Transforming3d(DataDecorator):
    """Transform a data

    Attributes:
        transformer (Transformer): Transform the data

    """
    def __init__(self, data, transformer, get_data_on_the_fly=True):
        super().__init__(data, get_data_on_the_fly)
        self.transformer = transformer

    def _get_data(self):
        """Transform the data
        
        Returns:
            transformed (numpy.array): The transformed data

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
    """Transform data using interpolation
    
    Attributes:
        transformer (Transformer): Transform the data
        order (int): Interpolation order; 0: nearest neighbor; 1: linear

    """
    def __init__(self, data, transformer, order=0, get_data_on_the_fly=True):
        super().__init__(data, transformer, get_data_on_the_fly)
        self.order = order

    def _get_data(self):
        """Transform the data
        
        Returns:
            transformed (numpy.array): The transformed data

        """
        data = self.transformer.transform(self.data.get_data(), self.order)
        return data


class Flipping3d(Transforming3d):
    """Flip data

    See .transformers.Flipper for more details. Flipping is also treated as a
    transform here, so additional Flipper is implemented to use the same API as
    Transforming3d.

    Attributes:
        label_pairs (list of list of int): Each element is a two-item list which
            contains the correspondence of the label to swap after flipping

    """
    def __init__(self, data, transformer, label_pairs=[],
                 get_data_on_the_fly=True):
        super().__init__(data, transformer, get_data_on_the_fly)
        self.label_pairs = label_pairs

    def _get_data(self):
        """Flip the data
        
        Returns:
            flipped (numpy.array): The flipped data

        """
        transformed = self.transformer.transform(self.data.get_data(),
                                                 self.label_pairs)
        return transformed
