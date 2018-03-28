# -*- coding: utf-8 -*-

import os
from image_processing_3d import crop3d, calc_bbox3d, resize_bbox3d
from image_processing_3d import rotate3d, deform3d, calc_random_deformation3d

from .data import Data, Data3d


class DataDecorator(Data):

    def __init__(self, data, get_data_on_the_fly=True):
        super().__init__(get_data_on_the_fly)
        self.data = data


class Cropping3d(DataDecorator):

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
        mask = self.mask.get_data()
        bbox = calc_bbox3d(mask)
        bbox = resize_bbox3d(bbox, self.cropping_shape)
        cropped, self._source_bbox, self._target_bbox = crop3d(data, bbox)
        return cropped


class Binarizing3d(DataDecorator):

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


class Transforming3d(DataDecorator):

    def __init__(self, data, transformer, get_data_on_the_fly=True):
        super().__init__(data, get_data_on_the_fly)
        self.transformer = transformer

    def update(self):
        self.transformer.update()
    
    def _get_data(self):
        return self.transformer.transform(self.data.get_data())


class Transformer:
    def update(self):
        raise NotImplementedError
    def transform(self, data):
        raise NotImplementedError


class Flipper(Transformer):

    def __init__(self, dim=1, label_pairs=[]):
        self.dim = dim
        self.label_pairs = label_pairs

    def update(self):
        pass

    def transform(self, data):
        flipped = np.flip(data, self.dim)
        for (pair1, pair2) in self.label_pairs:
            flipped[flipped==pair1] = pair2
            flipped[flipped==pair2] = pair1
        return flipped


class Rotator(Transformer):

    def __init__(self, max_angle=5, point=None, order=1):
        self.max_angle = max_angle
        self.point = point
        self.order = order

        self._rand_state = np.random.RandomState()
        self.update()

    def update(self):
        self._x_angle = self._calc_rand_angle()
        self._y_angle = self._calc_rand_angle()
        self._z_angle = self._calc_rand_angle()

    def transform(self, data):
        rotated = rotate3d(data, self._x_angle, self._y_angle, self._z_angle,
                           point=self.point, order=self.order)
        return rotated

    def _calc_rand_angle(self):
        angle = self._rand_state.rand(1)
        angle = float(angle * 2 * self.max_angle - self.max_angle)
        return angle


class Deformer(Transformer):

    def __init__(self, shape, sigma, scale, order=1):
        self.sigma = sigma
        self.scale = scale
        self.shape = shape
        self.order = order

        self._rand_state = np.random.RandomState()
        self.update()

    def update(self):
        self._x_deform = self._calc_random_deform()
        self._y_deform = self._calc_random_deform()
        self._z_deform = self._calc_random_deform()

    def transform(self, data):
        deformed = deform3d(data, self._x_deform, self._y_deform, self.z_deform,
                            self.order)
        return deformed

    def _calc_random_deform():
        scale = self._rand_state.rand(1) * self.scale
        deform = calc_random_deformation3d(self.shape, self.sigma, scale)
        return deform


class Composer(Transformer):

    def __init__(self, *transformers):
        self.transformers = transformers

    def update(self):
        for transformer in self.transformers:
            transformer.update()

    def transform(self, data):
        for transform in self.transformers:
            data = transform(data)
        return data
