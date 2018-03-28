# -*- coding: utf-8 -*-

import os
import numpy as np
from image_processing_3d import crop3d, calc_bbox3d, resize_bbox3d
from image_processing_3d import rotate3d, deform3d, calc_random_deformation3d

from .data import Data, Data3d


class DataDecorator(Data):
    """Abstract class to decorate Data

    Attributes:
        data (Data): The decorated data

    """
    def __init__(self, data, get_data_on_the_fly=True):
        super().__init__(get_data_on_the_fly)
        self.data = data

    def update(self):
        """Update the state/parameters"""
        self.data.update()


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
        mask = self.mask.get_data()
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
    """Transform a data

    Attributes:
        transformer (Transformer): Transform the data

    """
    def __init__(self, data, transformer, get_data_on_the_fly=True):
        super().__init__(data, get_data_on_the_fly)
        self.transformer = transformer

    def _get_data(self):
        """Transform the data"""
        return self.transformer.transform(self.data.get_data())

    def update(self):
        """Update the state/parameters"""
        super().update()
        self.transformer.update()


class Transformer:
    """Abstract class to transform data

    """  
    def update(self):
        """Abstract method to update parameters"""
        raise NotImplementedError

    def transform(self, data):
        """Abstract method to transform the data
        
        Args:
            data (numpy.array): The data to transform

        """
        raise NotImplementedError
    def share(self, *transformers):
        """Abstract to share parameters with other transformers
        
        Args:
            transformer (Transformer): The transformer to share parameteres
                with.

        """
        raise NotImplementedError


class Flipper(Transformer):
    """Flip the data along an dimension (axis)

    If `self.label_pairs` is empty, the class just flip the data; if not, the
    class which swap the corresponding labels after the flipping. For example,
    suppose 23 is a label on the left side of brain, while 26 on the right.
    After flipping the image, the labels 23 and 26 should be swapped to enforce
    they are at the correct sides.

    Attributes:
        dim (int): The dimension/axis that the data is flipped along
        label_pairs (list of list of int): Each element is a two-item list which
            contains the correspondence of the label to swap after flipping

    """
    def __init__(self, dim=1, label_pairs=[]):
        self.dim = dim
        self.label_pairs = label_pairs

    def update(self):
        pass

    def transform(self, data):
        """Flip the data

        Args:
            data (numpy.array): The data to flip

        Returns:
            flipped (numpy.array): The flipped data

        """
        flipped = np.flip(data, self.dim).copy()
        for (pair1, pair2) in self.label_pairs:
            mask1 = flipped==pair1
            mask2 = flipped==pair2
            flipped[mask1] = pair2
            flipped[mask2] = pair1
        return flipped


class Rotator(Transformer):
    """Rotate the data randomly

    Call `image_processing_3d.rotate3d` to rotate the data. The rotation angles
    are randomly sampled from a uniform distribution between `-self.max_angel`
    and `self.max_angle`. Each time `self.update()` is called, the rotation
    angles are resampled.

    Attributes:
        max_angle (positive int): Specify the sampling uniform distribution. In
            degrees.
        point ((3,) numpy.array): The point to rotate around
        order (int): Interpolation order. 0: nearest interpolation; 1: linear
            interpolation. Higher orders are not recommended since scipy'
            incorrect implementation (?).
        _rand_state (numpy.random.RandomState): Random sampling
        _x_angle, _y_angle, _z_angle ((1,) list of float): Rotation angles
            aroung x, y, and z axes. Use list so we can copy the vairable by
            refernce.

    """
    def __init__(self, max_angle=5, point=None, order=1):
        self.max_angle = max_angle
        self.point = point
        self.order = order

        self._rand_state = np.random.RandomState()
        self._x_angle = [0]
        self._y_angle = [0]
        self._z_angle = [0]
        self.update()

    def update(self):
        """Resample the rotation angles

        """
        self._x_angle[0] = self._calc_rand_angle()
        self._y_angle[0] = self._calc_rand_angle()
        self._z_angle[0] = self._calc_rand_angle()

    def transform(self, data):
        """Rotate the data

        Args:
            data (numpy.array): The data to rotate

        Returns:
            rotated (numpy.array): The rotated data
        
        """
        rotated = rotate3d(data, self._x_angle[0], self._y_angle[0],
                           self._z_angle[0], point=self.point, order=self.order)
        return rotated

    def _calc_rand_angle(self):
        """Calculate random angle from a uniform distribution
        
        Returns:
            angle (float): The sampled rotation angle

        """
        angle = self._rand_state.rand(1)
        angle = float(angle * 2 * self.max_angle - self.max_angle)
        return angle

    def share(self, *transformers):
        """Share rotation angles with other rotators

        Args:
            transformer (Rotator): The rotater to share parameters with

        """
        for transformer in transformers:
            transformer._x_angle = self._x_angle
            transformer._y_angle = self._y_angle
            transformer._z_angle = self._z_angle


class Deformer(Transformer):
    """Deform the data using elastic deformation
    
    Call external `image_processing_3d.deform3d` to perform the elastic
    transform. It creates a random deformation field specified by `self.sigma`
    (for deformation field smoothness) and `self.scale` (for displacement
    maginitude). The dispacement scale is randomly sampled from a uniform
    distribution [0, `self.scale`].

    Call `self.update()` to resample the deformation field.

    Attributes:
        shape (tuple of int): The shape of the data to deform
        sigma (float): Control the smoothness of the deformation field. The
            larger the value, the smoother the field
        scale (float): Control the magnitude of the displacement. In pixels,
            i.e. the larget displacement at a pixel along a direction is
            `self.scale`.
        order (float): Interpolation order. 0: nearest neighbor interp. 1:
            linear. Higher orders are not supported
        _rand_state (numpy.random.RandomState): Random sampler
        _x_deform, y_deform, z_deform (numpy.array) Pixelwise translation
            (deformation field) along x, y, and z axes.

    """
    def __init__(self, shape, sigma, scale, order=1):
        self.shape = shape
        self.sigma = sigma
        self.scale = scale
        self.order = order

        self._rand_state = np.random.RandomState()
        self._x_deform = [None]
        self._y_deform = [None]
        self._z_deform = [None]
        self.update()

    def update(self):
        """Resample the deformation field"""
        self._x_deform[0] = self._calc_random_deform()
        self._y_deform[0] = self._calc_random_deform()
        self._z_deform[0] = self._calc_random_deform()

    def transform(self, data):
        """Deform the data
        
        Args:
            data (numpy.array): The data to deform

        Returns:
            deformed (numpy.array): The deformed data

        """
        deformed = deform3d(data, self._x_deform[0], self._y_deform[0],
                            self._z_deform[0], self.order)
        return deformed

    def _calc_random_deform(self):
        """Randomly sample deformation (single axis)
        
        Returns:
            deform (numpy.array): The deformation field along a axis

        """
        scale = self._rand_state.rand(1) * self.scale
        deform = calc_random_deformation3d(self.shape, self.sigma, scale)
        return deform

    def share(self, *transformers):
        """Share deformation with other deformers

        Args:
            transformer (Deformer): The deformer to share parameters with

        """
        for transformer in transformers:
            transformer._x_deform = self._x_deform
            transformer._y_deform = self._y_deform
            transformer._z_deform = self._z_deform
