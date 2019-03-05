# -*- coding: utf-8 -*-

"""Implement Worker to process Image

"""
import numpy as np
from enum import Enum, auto
from image_processing_3d import rotate3d, scale3d
from py_singleton import Singleton

from .configs import Config
from .images import Mask


class WorkerName(Enum):
    flipping = auto()
    translation = auto()
    rotation = auto()
    scaling = auto()
    deformation = auto()
    cropping = auto()


class WorkerType(Enum):
    aug = auto()
    addon = auto()


class WorkerTypeMapping(metaclass=Singleton):
    """Map the worker name and its type

    Attributes:
        _mapping (dict): Internal strurcture keeping the mapping

    """
    def __init__(self):
        config = Config()
        if not set(config.total_addon).isdisjoint(config.total_aug):
            raise RunTimeError('Addon and aug workers overlap in config')
        self._mapping = {WorkerName[worker_name]: WorkerType.addon
                         for worker_name in config.total_addon}
        self._mapping.update({WorkerName[worker_name]: WorkerType.aug
                              for worker_name in config.total_aug})

    def __getitem__(self, worker):
        if type(worker) is str:
            worker = WorkerName[worker]
        return self._mapping[worker]

    def items(self):
        return self._mapping.items()


def create_worker(worker_name):
    """Create concrete worker to process images

    Args:
        worker_name (enum .workers.WorkerName): The name of the worker

    Returns:
        worker (.workers.Worker): The worker instance

    Raises:
        ValueError: The worker_name is not in enum WorkerName

    """
    if worker_name is WorkerName.translation:
        return Translator()
    elif worker_name is WorkerName.rotation:
        return Rotator()
    elif worker_name is WorkerName.scaling:
        return Scaler()
    elif worker_name is WorkerName.deformation:
        return Deformer()
    elif worker_name is WorkerName.flipping:
        return Flipper()
    elif worker_name is WorkerName.cropping:
        return Cropper()
    else:
        raise ValueError('Worker "%s" is not in WorkerName')


class Worker:
    """Abstract class to process .images.Image
    
    """
    message = ''

    def __init__(self):
        pass

    def process(self, *images):
        """Process a set of .images.Image instances

        Args:
            image (.images.Image): The image to process

        Returns:
            results (tuple of .images.Image): The processed images
        
        """
        results = list()
        for image in images:
            data = self._process(image)
            results.append(image.update(data, self.message))
        return tuple(results)

    def _process(self, image):
        """Abstract method to process an .image.Image instance

        Args:
            image (.image.Image): The image to process

        Returns:
            result (numpy.array): The processed image data
        
        """
        raise NotImplementedError


class Rotator(Worker):
    """Rotate Image randomly

    Call `image_processing_3d.rotate3d` to rotate the data. The rotation angles
    are randomly sampled from a uniform distribution between `-self.max_angel`
    and `self.max_angle`.

    Attributes:
        max_angle (int): Specify the sampling uniform distribution in degrees
        point (numpy.array): The 3D point to rotate around
        _rand_state (numpy.random.RandomState): Specify random seed
        _x (float): Rotation angle aroung the x axis in degrees
        _y (float): Rotation angle aroung the y axis in degrees
        _z (float): Rotation angle aroung the z axis in degrees

    """
    message = 'rotate'

    def __init__(self, max_angle=5, point=None):
        """Initialize

        """
        self.max_angle = max_angle
        self.point = point

        self._rand_state = np.random.RandomState()
        self._x = self._calc_rand_angle()
        self._y = self._calc_rand_angle()
        self._z = self._calc_rand_angle()

    def _calc_rand_angle(self):
        """Calculate random angle from a uniform distribution
        
        Returns:
            angle (float): The sampled rotation angle in degrees

        """
        angle = self._rand_state.rand(1)
        angle = float(angle * 2 * self.max_angle - self.max_angle)
        return angle

    def _process(self, image):
        """Rotate an image

        Args:
            image (.image.Image): The image to rotate

        Returns:
            result (numpy.data): The rotated image

        """
        return rotate3d(image.data, self._x, self._y, self._z,
                        point=self.point, order=image.interp_order)


class Scaler(Worker):
    """Scale Image randomly

    Call `image_processing_3d.scale3d` to scale the data. The scaling factors
    are randomly sampled from a uniform distribution between 1
    and `self.max_scale` and between -1 and -self.max_scale. If the sampled is
    negative, convert to 1/abs(scale).

    Attributes:
        max_scale (float): Specify the sampling uniform distribution.
        point (numpy.array): The 3D point to scale around
        _rand_state (numpy.random.RandomState): Specify random seed
        _x (float): Scaling factor around the x axis
        _y (float): Scaling factor around the y axis
        _z (float): Scaling factor around the z axis

    """
    message  = 'scale'

    def __init__(self, max_scale=2, point=None):
        """Initialize

        """
        self.max_scale = max_scale
        self.point = point

        self._rand_state = np.random.RandomState()
        self._x = self._calc_rand_scale()
        self._y = self._calc_rand_scale()
        self._z = self._calc_rand_scale()

    def _calc_rand_scale(self):
        """Calculate random scaling factor from a uniform distribution
        
        Returns:
            scale (float): The sampled scaling factor

        """
        scale = self._rand_state.rand(1)
        scale = float(scale * (self.max_scale - 1) + 1)
        if self._rand_state.choice([-1, 1]) < 0:
            scale = 1 / scale
        return scale

    def _process(self, image):
        """Scale an image

        Args:
            image (.images.Image): The image to scale

        Returns:
            result (numpy.data): The scaled image data
        
        """
        return scale3d(image.data, self._x, self._y, self._z,
                       point=self.point, order=image.interp_order)


class Flipper(Worker):
    """Flip the image along an dimension (axis)

    Attributes:
        dim (int): The dimension/axis that the image is flipped around

    """
    message = 'flip'

    def __init__(self, dim=0):
        self.dim = dim

    def _process(self, image):
        """Flip the image

        Only flip the data if image does not have `label_pairs`; otherwise, the
        corresponding labels are swapped after the flipping. For example,
        suppose 23 is a label on the left side of brain, while 26 on the right.
        After flipping the image, the labels 23 and 26 should be swapped so they
        are on the correct sides

        Args:
            image (.image.Image): The image to flip

        Returns:
            result (numpy.array): The flipped image data

        """
        result = np.flip(image.data, self.dim).copy()
        if hasattr(image, 'pairs'):
            for (pair1, pair2) in image.pairs:
                mask1 = result==pair1
                mask2 = result==pair2
                result[mask1] = pair2
                result[mask2] = pair1
        return result


class Cropper(Worker):
    """Crop .images.Image instances with .images.Mask

    TODO:
        update results as an 2D array for multiple masks

    """
    def process(self, *images):
        """Crop the images

        If any images of class .images.Mask will be used to crop the rest of
        images

        Arags:
            image (.images.Image): The image to crop or the mask

        Returns:
            results (list of 

        """
        masks = list()
        others = list()
        for image in images:
            if isinstance(image, Mask):
                masks.append(image)
            else:
                others.append(image)

        results = list()
        for mask in masks:
            cropped = list()
            for image in others:
                cropped.append(mask.crop(image))
            # results.append(cropped) # TODO
            results.extend(cropped)
        return results
