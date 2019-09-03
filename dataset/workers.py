# -*- coding: utf-8 -*-

"""Workers to process images.

"""
from enum import Enum, auto
import time
import numpy as np
from image_processing_3d import rotate3d, scale3d, padcrop3d, translate3d_int
from image_processing_3d import calc_random_deformation3d, deform3d, crop3d
from image_processing_3d import calc_random_intensity_transform as calc_int

from .config import Config
from .images import Mask, Label, Image


class WorkerType(Enum):
    """The type of a worker.

    Attributes:
        AUG: For workers used in augmentation.
        ADDON: For workers used in additional processing.

    """
    AUG = auto()
    ADDON = auto()


class WorkerCreator:
    """Creates a concerete worker to process the images.

    Call the method :meth:`register` to add a new :class:`Worker` into the pool
    so when calling the method :meth:`create` with a worker name (:class:`str`)
    to create an worker, it can find the corresponding implementation correctly.
    This feature is primarily for adding new type of workers. Call the method
    :meth:`show` to print out all registered workers.

    Note:
        The registered workers are stored as a class attribute and all methods
        of this class are class methods. Use this class without initialization.

    Note:
        The worker type (AUG or ADDON) is changed via
        :class:`dataset.config.Config` and take effect when calling the method
        :meth:`register`.

    """
    _workers = dict()

    @classmethod
    def create(cls, name):
        """Creates a worker.

        Args:
            name (str): The name of the worker. Call the method :meth:`register`
                to register new workers.
        
        Returns:
            :class:`Worker`: a concrete worker.

        Raises:
            KeyError: The name is not registered.
            
        """
        if name in cls._workers:
            return cls._workers[name]()
        else:
            raise KeyError('Worker "%s" is not registered.' % name)

    @classmethod
    def register(cls, name, worker):
        """Registers an instance of :class:`Worker` with a :class:`str`.

        Args:
            name (str): The name of the worker.
            worker (type): The class :class:`Worker` to register.

        Raises:
            RuntimeError: Worker types are incorrect in config.
        
        """
        if cls._config_is_correct():
            message = 'Addon and augmentation workers overlap in configurations'
            raise RuntimeError(message)
        if name in Config.worker_types['addon']:
            worker.worker_type = WorkerType.ADDON
        elif name in Config.worker_types['aug']:
            worker.worker_type = WorkerType.AUG
        else:
            raise RuntimeError('Worker %s is not in the config.' % name)
        cls._workers[name] = worker

    @staticmethod
    def _config_is_correct():
        """Returns if aug and addon workers are correctly written in config."""
        addon = Config.worker_types['addon']
        aug = Config.worker_types['aug']
        return not set(addon).isdisjoint(aug)

    @classmethod
    def unregister(cls, name):
        """Unregisters a worker."""
        cls._workers.pop(name)

    @classmethod
    def get_type(cls, name):
        """Returns the type (AUG or ADDON) of worker."""
        return cls._workers[name].worker_type

    @classmethod
    def show(cls):
        """Prints out the registered workers."""
        message = ['Registered workers:']
        length = max([len(k) for k in cls._workers.keys()])
        for k, v in cls._workers.items():
            t = WorkerType(v.worker_type)
            message.append(('    %%%ds: %%s, %%s' % length) % (k, v, t.name))
        return '\n'.join(message)


class Worker:
    """Abstract class to process instances of :class:`dataset.images.Image`.
    
    Attributes:
        message (str): The message to show when printing.
        worker_type (WorkerType, enum): The type (AUG or ADDON) of the worker.
    
    """
    message = ''
    worker_type = WorkerType.ADDON

    def process(self, *images):
        """Processes instances of :class:`dataset.images.Image`.

        Args:
            image (dataset.images.Image): The image to process.

        Returns:
            tuple[dataset.images.Image]: The processed images.
        
        """
        results = list()
        for image in images:
            data = self._process(image)
            results.append(image.update(data, self.message))
        return tuple(results)

    def _process(self, image):
        """Processes an :class:`dataset.image.Image` instance.

        Args:
            image (dataset.images.Image): The image to process.

        Returns:
            numpy.ndarray: The processed image data.
        
        """
        raise NotImplementedError


class RandomWorker(Worker):
    """Abstract class for a worker with operations sampled randomly.
   
    Attributes:
        rand_state (numpy.random.RandomState): The random state.
    
    """
    def __init__(self):
        self.rand_state = np.random.RandomState(int(time.time()))


class Resizer_(Worker):
    """Resizes images by padding or cropping.

    Attributes:
        shape (iterable[int]): The target image shape.

    """
    message = 'resize'
    worker_type = WorkerType.ADDON

    def __init__(self, shape):
        self.shape = shape

    def _process(self, image):
        return padcrop3d(image.data, self.shape)[0]


class Resizer(Resizer_):
    """Wrapper of :class:`Resizer_`."""
    def __init__(self):
        super().__init__(shape=Config.image_shape)


class Rotator_(RandomWorker):
    """Rotates images randomly.

    The rotation angles are randomly sampled from a uniform distribution
    :math:`[-\\alpha, \\alpha]`, :math:`\\alpha` is specified by the attribute
    :attr:`max_angle`.
    .

    Attributes:
        max_angle (int): Specifies the sampling uniform distribution in degrees.
        point (numpy.ndarray): The 3D point to rotate around.
        x (float): The sampled rotation angle around the x axis.
        y (float): The sampled rotation angle around the y axis.
        z (float): The sampled rotation angle around the z axis.

    """
    message = 'rotate'
    worker_type = WorkerType.AUG

    def __init__(self, max_angle=5, point=None):
        super().__init__()
        self.max_angle = max_angle
        self.point = point
        self.x = self._calc_rand_angle()
        self.y = self._calc_rand_angle()
        self.z = self._calc_rand_angle()

    def _calc_rand_angle(self):
        """Returns a sampled angle in degrees."""
        angle = self.rand_state.rand(1)
        angle = float(angle * 2 * self.max_angle - self.max_angle)
        return angle

    def _process(self, image):
        return rotate3d(image.data, self.x, self.y, self.z,
                        pivot=self.point, order=image.interp_order)


class Rotator(Rotator_):
    """Wrapper of :class:`Rotator_`."""
    def __init__(self):
        super().__init__(max_angle=Config.max_rot_angle)


class Scaler_(RandomWorker):
    """Scales images randomly.

    The scaling factors are randomly sampled from uniform distributions
    :math:`[1, s]` and :math:`[-s, -1]`. :math:`s` is specified by the attribute
    :attr:`max_scale`. If the sampled scaling factor :math:`v` is negative,
    convert it to :math:`1/|v|`.

    Attributes:
        max_scale (float): Specifies the sampling uniform distribution.
        point (numpy.ndarray): The 3D point to scale around.
        x (float): The scaling factor around the x axis.
        y (float): The scaling factor around the y axis.
        z (float): The scaling factor around the z axis.

    """
    message  = 'scale'
    worker_type = WorkerType.AUG

    def __init__(self, max_scale=2, point=None):
        super().__init__()
        self.max_scale = max_scale
        self.point = point
        self.x = self._calc_rand_scale()
        self.y = self._calc_rand_scale()
        self.z = self._calc_rand_scale()

    def _calc_rand_scale(self):
        """Returns a random scaling factor from a uniform distribution."""
        scale = self.rand_state.rand(1)
        scale = float(scale * (self.max_scale - 1) + 1)
        if self.rand_state.choice([-1, 1]) < 0:
            scale = 1 / scale
        return scale

    def _process(self, image):
        return scale3d(image.data, self.x, self.y, self.z,
                       pivot=self.point, order=image.interp_order)


class Scaler(Scaler_):
    """Wrapper of :class:`Scaler_`."""
    def __init__(self):
        super().__init__(max_scale=Config.max_scale)


class Flipper_(Worker):
    """Flips images along an dimension/axis.

    Flip only the data if the image does NOT have the attribute ``pairs`` as
    :attr:`dataset.Label.pairs`. Otherwise, the corresponding labels are
    swapped after the flipping. For example, suppose 23 is a label on the left
    side of brain, while 26 is on the right. After flipping the image, the
    labels 23 and 26 should be swapped so they would appear on the correct
    sides.

    Attributes:
        dim (int): The dimension/axis that the image is flipped around.

    """
    message = 'flip'
    worker_type = WorkerType.ADDON

    def __init__(self, dim=0):
        self.dim = dim

    def _process(self, image):
        result = np.flip(image.data, self.dim).copy()
        if hasattr(image, 'pairs'):
            for (pair1, pair2) in image.pairs:
                mask1 = result==pair1
                mask2 = result==pair2
                result[mask1] = pair2
                result[mask2] = pair1
        return result


class Flipper(Flipper_):
    """Wrapper of :class:`Flipper_`."""
    def __init__(self):
        super().__init__(dim=Config.flip_dim)


class Cropper(Worker):
    """Crops images with a mask.

    Note:
        The last :class:`Mask` instance will be used to crop others, and all 
        :class:`Mask` instances will be removed from the outputs.

    """
    message = 'crop'
    worker_type = WorkerType.ADDON

    def process(self, *images):
        others = list()
        for image in images:
            if isinstance(image, Mask):
                mask = image
            else:
                others.append(image)
        results = tuple([mask.crop(image) for image in others])
        return results


class Translator_(RandomWorker):
    """Translates images randomly along x, y, and z axes by integers.

    The translations are randomly sampled from a uniform distribution
    :math:`[-t, t]` where :math:`t` is specified by the attribute
    attr:`max_trans`.

    Attributes:
        max_trans (int): Specifies the distribution.
        x (int): The translation along x axis.
        y (int): The translation along y axis.
        z (int): The translation along z axis.

    """
    message = 'translate'
    worker_type = WorkerType.AUG

    def __init__(self, max_trans=30):
        super().__init__()
        self.max_trans = max_trans
        self.x = self._calc_random_trans()
        self.y = self._calc_random_trans()
        self.z = self._calc_random_trans()

    def _process(self, image):
        return translate3d_int(image.data, self.x, self.y, self.z)

    def _calc_random_trans(self):
        """Returns the sampled translation."""
        trans = self.rand_state.rand(1)
        trans = int(np.round(trans * 2 * self.max_trans - self.max_trans))
        return trans


class Translator(Translator_):
    """Wrapper of :class:`Translator_`."""
    def __init__(self):
        super().__init__(max_trans=Config.max_trans)


class Deformer_(RandomWorker):
    """Deforms the images randomly.
    
    It creates a random deformation field specified by the attribute
    :attr:`sigma`, for deformation field smoothness, and the attribute
    :attr:`scale`, for displacement maginitude. See
    :func:`image_processing_3d.deform3d` for more information.

    Note:
        The method :meth:`process` does not know the shape of the image in
        advance, so it cannot sample the deformation fields during
        initialization. This class also does not keep references to the
        deformation fileds to save memory.

    Attributes:
        sigma (float): Controls the smoothness of the deformation field. The
            larger the value, the smoother the field
        scale (float): Controls the magnitude of the displacement in voxels,
            i.e. this value is the larget displacement at a voxel along an axis.
        rand_state (numpy.random.RandomState): The random state.
        
    """
    message = 'deform'
    worker_type = WorkerType.AUG

    def __init__(self, sigma, scale):
        self.sigma = sigma
        self.scale = scale
        self.rand_state = np.random.RandomState(int(time.time()))

    def process(self, *images):
        shape = images[0].shape
        x_deform = self._calc_random_deform(shape)
        y_deform = self._calc_random_deform(shape)
        z_deform = self._calc_random_deform(shape)
        results = list()
        for image in images:
            data = deform3d(image.data, x_deform, y_deform, z_deform,
                            order=image.interp_order)
            results.append(image.update(data, self.message))
        return tuple(results)

    def _calc_random_deform(self, shape):
        """Samples the deformation along a single axis.

        Args:
            shape (tuple): The shape of the image to apply the deformation to.
        
        Returns:
            numpy.ndarray: The deformation field along a axis.

        """
        scale = self.rand_state.rand(1) * self.scale
        deform = calc_random_deformation3d(shape, self.sigma, scale)
        return deform


class Deformer(Deformer_):
    """Wrapper of :class:`Deformer_`."""
    def __init__(self):
        super().__init__(sigma=Config.def_sigma, scale=Config.def_scale)


class LabelNormalizer(Worker):
    """Normalizes label images to replace label values with ``0 : num_labels``.

    Note:
        This operation are only applied to :class:`dataset.images.Label` by
        calling its method :meth:`dataset.Images.Label.norm`.

    """
    message = 'norm_label'
    worker_type = WorkerType.ADDON

    def process(self, *images):
        results = list()
        for image in images:
            if isinstance(image, Label):
                results.append(image.normalize())
            else:
                results.append(image)
        return tuple(results)


class MaskExtractor_(Worker):
    """Extracts a mask from a label value.

    Note:
        This class only affects instances of :class:`dataset.images.Label`.

    Args:
        label_value (int): The label value to extract the mask.

    """
    message = 'extract_mask'
    worker_type = WorkerType.ADDON

    def __init__(self, label_value):
        self.label_value = label_value

    def _process(self, image):
        data = image.data
        if isinstance(image, Label):
            return (data == self.label_value).astype(data.dtype)
        else:
            return data


class MaskExtractor(MaskExtractor_):
    """Wrapper of :class:`MaskExtractor_`."""
    def __init__(self):
        super().__init__(Config.mask_label_val)


class PatchExtractor_(RandomWorker):
    """Extracts patches from an image randomly.

    The patch should be within the image; therefore, the smallest possible start
    index is 0, and the largest possible start is ``image_shape`` -
    ``patch_shape``. The start is uniformly sampled.

    Note:
        This worker should normally be put into the end of the pipeline and
        should use :func:`patch_collate` to collate the samples for
        :class:`torch.utils.data.DataLoader`.

    Attributes:
        patch_shape (iterable[int]): The 3D spatial shape of the patch.
        num_patches (int): The number of patches to extract.

    """
    message = 'extract_patches'
    worker_type = WorkerType.ADDON

    def __init__(self, patch_shape=(10, 10, 10), num_patches=1):
        super().__init__()
        self.patch_shape = patch_shape
        self.num_patches = num_patches

    def process(self, *images):
        results = list()
        image_shape = images[0].shape[-3:]
        for i in range(self.num_patches):
            # TODO: without replacement
            patch_bbox = self._calc_patch_bbox(image_shape)
            for image in images:
                data = crop3d(image.data, patch_bbox)[0]
                message = '%s_%d' % (self.message, i)
                results.append(image.update(data, message))
        return results

    def _calc_patch_bbox(self, image_shape):
        """Returns the bounding box of a patch."""
        bbox = list()
        for ims, ps in zip(image_shape, self.patch_shape):
            lower_bound = 0
            upper_bound = ims - ps
            start = self.rand_state.choice(np.arange(lower_bound, upper_bound))
            bbox.append(slice(start, start + ps))
        return tuple(bbox)


class PatchExtractor(PatchExtractor_):
    """Wrapper of :class:`PatchExtractor_`."""
    def __init__(self):
        super().__init__(patch_shape=Config.patch_shape,
                         num_patches=Config.num_patches)


def _calc_transpose_axes(num_axes, slice_dim):
    axes = list(range(num_axes))
    dim = axes.pop(slice_dim)
    return [dim] + axes


class SliceExtractor_(RandomWorker):
    """Extracts slices from an image randomly.

    Note:
        This worker should normally be put into the end of the pipeline and
        should use :func:`dataset.funcs.patch_collate` to collate the samples
        for :class:`torch.utils.data.DataLoader`.

    Attributes:
        num_slices (int): The number of slices to extract.
        slice_dim (int): The slice dimension. Counting from the last dimension
            when negative.

    """
    message = 'extract_slices'
    worker_type = WorkerType.ADDON

    def __init__(self, num_slices=1, slice_dim=-1):
        super().__init__()
        self.num_slices = num_slices
        self.slice_dim = slice_dim

    def process(self, *images):
        results = list()
        shape = images[0].shape
        axes = _calc_transpose_axes(len(shape), self.slice_dim)
        indices = self._calc_slicing_indices(shape)
        for image in images:
            slices = image.data[indices]
            slices = np.transpose(slices, axes)
            results.append(image.update(slices, self.message))
        return results

    def _calc_slicing_indices(self, image_shape):
        total_slices = image_shape[self.slice_dim]
        num = total_slices if self.num_slices is None else self.num_slices
        indices = self.rand_state.choice(total_slices, num, replace=False)
        result = [slice(None)] * len(image_shape)
        result[self.slice_dim] = indices
        return tuple(result)


class SliceExtractor(SliceExtractor_):
    """Wrapper of :class:`SliceExtractor_`."""
    def __init__(self):
        super().__init__(num_slices=Config.num_slices,
                         slice_dim=Config.slice_dim)


class DimConverter_(Worker):
    """Converts a 3D image to multi-slice 2D images.

    Note:
        This worker should normally be put into the end of the pipeline and
        should use :func:`dataset.funcs.patch_collate` to collate the samples
        for :class:`torch.utils.data.DataLoader`.
    
    Attributes:
        slice_dim (int): The slice dimension. Counting from the last dimension
            when negative.

    """
    message = 'convert_dim'
    worker_type = WorkerType.ADDON

    def __init__(self, slice_dim):
        super().__init__()
        self.slice_dim = slice_dim

    def process(self, *images):
        results = list()
        shape = images[0].shape
        axes = _calc_transpose_axes(len(shape), self.slice_dim)
        for image in images:
            data = np.transpose(image.data, axes)
            results.append(image.update(data, self.message))
        return results


class DimConverter(DimConverter_):
    """Wrapper of :class:`DimConverter_`."""
    def __init__(self):
        super().__init__(slice_dim=Config.slice_dim)


WorkerCreator.register('resize', Resizer)
WorkerCreator.register('crop', Cropper)
WorkerCreator.register('norm_label', LabelNormalizer)
WorkerCreator.register('extract_mask', MaskExtractor)
WorkerCreator.register('extract_patches', PatchExtractor)
WorkerCreator.register('extract_slices', SliceExtractor)
WorkerCreator.register('convert_dim', DimConverter)
WorkerCreator.register('flip', Flipper)
WorkerCreator.register('rotate', Rotator)
WorkerCreator.register('deform', Deformer)
WorkerCreator.register('scale', Scaler)
WorkerCreator.register('translate', Translator)
