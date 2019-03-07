# -*- coding: utf-8 -*-

"""Implement Worker to process Image

"""
import numpy as np
from enum import Enum, auto
from py_singleton import Singleton
from image_processing_3d import rotate3d, scale3d
from image_processing_3d import calc_random_deformation3d, deform3d

from .configs import Config
from .images import Mask, Label


class WorkerName(Enum):
    flipping = auto()
    translation = auto()
    rotation = auto()
    scaling = auto()
    deformation = auto()
    cropping = auto()
    label_normalization = auto()


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
    config = Config()
    if worker_name is WorkerName.flipping:
        return Flipper(dim=config.flip_dim)
    elif worker_name is WorkerName.cropping:
        return Cropper()
    elif worker_name is WorkerName.label_normalization:
        return LabelNormalizer()
    elif worker_name is WorkerName.translation:
        return Translator(max_trans=config.max_trans)
    elif worker_name is WorkerName.rotation:
        return Rotator(max_angle=config.max_rot_angle)
    elif worker_name is WorkerName.scaling:
        return Scaler(max_scale=config.max_scale)
    elif worker_name is WorkerName.deformation:
        return Deformer(shape=config.image_shape, sigma=config.def_sigma,
                        scale=config.def_scale)
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
            results (tuple of .images.Image): The cropped images

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
        return tuple(results)


class Translator(Worker):
    """Translate images randomly along x, y, and z axes

    The translation is integer for simplicity
    
    Attributes:
        max_trans (int): The translation will be uniformly sampled from
            [-self.max_trans, self.max_trans]
        _rand_state (numpy.random.RandomState): Random sampling
        _x (int): Translation along x axis
        _y (int): Translation along y axis
        _z (int): Translation along z axis

    """
    message = 'translate'

    def __init__(self, max_trans=30):
        self.max_trans = max_trans
        self._rand_state = np.random.RandomState()
        self._x = self._calc_random_trans()
        self._y = self._calc_random_trans()
        self._z = self._calc_random_trans()

    def _process(self, image):
        """Translate an image
        
        Args:
            image (.image.Image): The image to translate

        Returns:
            result (numpy.data): The translated image

        """
        data = image.data
        result = np.zeros_like(data)
        xs, xt = self._calc_index(self._x, data.shape[0])
        ys, yt = self._calc_index(self._y, data.shape[1])
        zs, zt = self._calc_index(self._z, data.shape[2])
        result[..., xt, yt, zt] = data[..., xs, ys, zs]
        return result

    def _calc_index(self, trans, size):
        """Calculate target and source indexing slices from translation

        Args:
            trans (int): The translation of the data
            size (int): The size of the data

        Returns:
            source (slice): The indexing slice in the source data
            target (slice): The indexing slice in the target data

        """
        if trans > 0:
            source = slice(0, size-1-trans, None)
            target = slice(trans, size-1, None)
        elif trans <= 0:
            source = slice(-trans, size-1, None)
            target = slice(0, size-1+trans, None)
        return source, target

    def _calc_random_trans(self):
        """Randomly sample translation along an axis
        
        Returns:
            trans (int): The translation along an axis

        """
        trans = self._rand_state.rand(1)
        trans = int(np.round(trans * 2 * self.max_trans - self.max_trans))
        return trans


class Deformer(Worker):
    """Deform the images randomly
    
    Call external `image_processing_3d.deform3d` to perform the elastic
    transform. It creates a random deformation field specified by `self.sigma`
    (for deformation field smoothness) and `self.scale` (for displacement
    maginitude). The dispacement scale is randomly sampled from a uniform
    distribution [0, `self.scale`].

    Attributes:
        shape (tuple of int): The shape of the data to deform
        sigma (float): Control the smoothness of the deformation field. The
            larger the value, the smoother the field
        scale (float): Control the magnitude of the displacement. In pixels,
            i.e. the larget displacement at a pixel along a direction is
            `self.scale`.
        _rand_state (numpy.random.RandomState): Random sampler
        _x (numpy.array) Pixelwise translation (deformation field) along x axis
        _x (numpy.array) Pixelwise translation (deformation field) along y axis
        _x (numpy.array) Pixelwise translation (deformation field) along z axis

    """
    message = 'deform'

    def __init__(self, shape, sigma, scale):
        self.shape = shape
        self.sigma = sigma
        self.scale = scale
        self._rand_state = np.random.RandomState()
        self._x = self._calc_random_deform()
        self._y = self._calc_random_deform()
        self._z = self._calc_random_deform()

    def _process(self, image):
        """Deform an image
        
        Args:
            image (.image.Image): The image to deform

        Returns:
            result (numpy.array): The deformed image

        """
        return deform3d(image.data, self._x, self._y, self._z,
                        order=image.interp_order)

    def _calc_random_deform(self):
        """Randomly sample deformation (single axis)
        
        Returns:
            deform (numpy.array): The deformation field along a axis

        """
        scale = self._rand_state.rand(1) * self.scale
        deform = calc_random_deformation3d(self.shape, self.sigma, scale)
        return deform


class LabelNormalizer(Worker):
    """Normalize label image to replace label values with 0 : num_labels

    """
    def process(self, *images):
        """Normalize the label images

        Only affect .images.Label instances

        Args:
            image (.images.Image): The image to normalize

        Returns:
            results (tuple of .images.Image): The normalized images
            
        """
        results = list()
        for image in images:
            if isinstance(image, Label):
                results.append(image.normalize())
            else:
                results.append(image)
        return tuple(results)
