# -*- coding: utf-8 -*-

import numpy as np
from image_processing_3d import rotate3d, deform3d, calc_random_deformation3d


class Transformer:
    """Abstract class to transform data

    """  
    def update(self):
        """Abstract method to update parameters"""
        raise NotImplementedError

    def cleanup(self):
        """Abstract method to cleanup transformer's attributes to save memory"""
        raise NotImplementedError

    def transform(self, data):
        """Abstract method to transform the data
        
        Args:
            data (numpy.array): The data to transform

        """
        raise NotImplementedError


class Interpolater(Transformer):
    """Abstract class to transform data using interpolation

    """
    def transform(self, data, order):
        """Abstract method to transform the data using interpolation

        Args:
            data (numpy.array): The data to transform
            order (int): The interpolation order. 0: nearest neighbor
                interpolation; 1: linear interpolation

        """
        raise NotImplementedError


class Flipper(Transformer):
    """Flip the data along an dimension (axis)

    Attributes:
        dim (int): The dimension/axis that the data is flipped along

    """
    def __init__(self, dim=1):
        self.dim = dim

    def update(self):
        pass

    def cleanup(self):
        pass

    def transform(self, data, label_pairs=[]):
        """Flip the data

        If `label_pairs` is empty, the class just flip the data; if not, the
        class which swap the corresponding labels after the flipping. For
        example, suppose 23 is a label on the left side of brain, while 26 on
        the right.  After flipping the image, the labels 23 and 26 should be
        swapped to enforce they are at the correct sides.

        Args:
            data (numpy.array): The data to flip
            label_pairs (list of list of int): Each element is a two-item list
                which contains the correspondence of the label to swap after
                flipping

        Returns:
            flipped (numpy.array): The flipped data

        """
        flipped = np.flip(data, self.dim).copy()
        for (pair1, pair2) in label_pairs:
            mask1 = flipped==pair1
            mask2 = flipped==pair2
            flipped[mask1] = pair2
            flipped[mask2] = pair1
        return flipped


class Rotator(Interpolater):
    """Rotate the data randomly

    Call `image_processing_3d.rotate3d` to rotate the data. The rotation angles
    are randomly sampled from a uniform distribution between `-self.max_angel`
    and `self.max_angle`. Each time `self.update()` is called, the rotation
    angles are resampled.

    Attributes:
        max_angle (positive int): Specify the sampling uniform distribution. In
            degrees.
        point ((3,) numpy.array): The point to rotate around
        _rand_state (numpy.random.RandomState): Random sampling
        _x_angle, _y_angle, _z_angle ((1,) list of float): Rotation angles
            aroung x, y, and z axes. Use list so we can copy the vairable by
            refernce.

    """
    def __init__(self, max_angle=5, point=None):
        self.max_angle = max_angle
        self.point = point

        self._rand_state = np.random.RandomState()

    def update(self):
        """Resample the rotation angles

        """
        self._x_angle = self._calc_rand_angle()
        self._y_angle = self._calc_rand_angle()
        self._z_angle = self._calc_rand_angle()

    def cleanup(self):
        pass

    def transform(self, data, order):
        """Rotate the data

        Args:
            data (numpy.array): The data to rotate
            order (int): The interpolation order

        Returns:
            rotated (numpy.array): The rotated data
        
        """
        rotated = rotate3d(data, self._x_angle, self._y_angle, self._z_angle,
                           point=self.point, order=order)
        return rotated

    def _calc_rand_angle(self):
        """Calculate random angle from a uniform distribution
        
        Returns:
            angle (float): The sampled rotation angle

        """
        angle = self._rand_state.rand(1)
        angle = float(angle * 2 * self.max_angle - self.max_angle)
        return angle


class Deformer(Interpolater):
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
        _rand_state (numpy.random.RandomState): Random sampler
        _x_deform, y_deform, z_deform (numpy.array) Pixelwise translation
            (deformation field) along x, y, and z axes.

    """
    def __init__(self, shape, sigma, scale):
        self.shape = shape
        self.sigma = sigma
        self.scale = scale

        self._rand_state = np.random.RandomState()

    def update(self):
        """Resample the deformation field"""
        self._x_deform = self._calc_random_deform()
        self._y_deform = self._calc_random_deform()
        self._z_deform = self._calc_random_deform()

    def cleanup(self):
        """Set _deform to None to save memory"""
        self._x_deform = None
        self._y_deform = None
        self._z_deform = None

    def transform(self, data, order):
        """Deform the data
        
        Args:
            data (numpy.array): The data to deform
            order (int): Interpolation order

        Returns:
            deformed (numpy.array): The deformed data

        """
        deformed = deform3d(data, self._x_deform, self._y_deform,
                            self._z_deform, order=order)
        return deformed

    def _calc_random_deform(self):
        """Randomly sample deformation (single axis)
        
        Returns:
            deform (numpy.array): The deformation field along a axis

        """
        scale = self._rand_state.rand(1) * self.scale
        deform = calc_random_deformation3d(self.shape, self.sigma, scale)
        return deform
