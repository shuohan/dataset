# -*- coding: utf-8 -*-

import os
from image_processing_3d import crop3d, calc_bbox3d, resize_bbox3d

from .data import Data, Data3d


class DataDecorator(Data):

    def __init__(self, data, get_data_on_the_fly=True):
        super().__init__(get_data_on_the_fly)
        self.data = data


class Cropping3d(DataDecorator):

    def __init__(self, data, cropping_shape, get_data_on_the_fly=True):
        super().__init__(data, get_data_on_the_fly)
        self.cropping_shape = cropping_shape
        self.mask = Data3d(self._get_mask_filepath(), self.get_data_on_the_fly,
                           self.data.transpose4d)

        self._source_bbox = None
        self._target_bbox = None

    def _get_mask_filepath(self):
        if self.data.filepath.endswith('.nii.gz'):
            filename, ext = self.data.filepath.replace('.nii.gz', ''), '.nii.gz'
        else:
            filename, ext = os.path.splitext(self.data.filepath)
        tmp = filename.split('_')
        tmp[-1] = 'mask'
        mask_filepath = '_'.join(tmp) + ext
        return mask_filepath

    def _get_data(self):
        """Crop the data using the corresponding mask

        Returns:
            cropped (num_channels x num_i x ... numpy.array): The cropped data

        """
        data = self.data.get_data()
        mask = self.mask.get_data()
        bbox = calc_bbox3d(mask)
        bbox = resize_bbox3d(bbox, self.cropping_shape)
        cropped, self._source_bbox, self._target_bbox = crop3d(data, bbox)
        return cropped


class Binarizing3d(Dataset3dDecorator):

    def __init__(self, data, binarizer, get_data_on_the_fly=True):
        super().__init__(self, data, get_data_on_the_fly)
        self.binarizer = binarizer

    def _get_data(self):
        """Binarize the label image

        Fit the binarizer if not fitted before, and transform the label image.
        Since binarizer returns result with channels last, move the channels
        first.

        Returns:
            binarized (num_channels x num_i ... numpy.array): The binarized
                label image

        """
        data = self.data.get_data()
        if not hasattr(self.binarizer, 'classes_'):
            self.binarizer.fit(np.unique(data))
        binarized = self.binarizer.transform(data)
        binarized = np.rollaxis(binarized, -1) # channels first
        return binarized
