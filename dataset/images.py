# -*- coding: utf-8 -*-

"""Implement Image to handle data

"""
import os
import numpy as np
from glob import glob
from collections import defaultdict
from image_processing_3d import calc_bbox3d, resize_bbox3d, crop3d
from py_singleton import Singleton

from .config import Config
from .loads import load, load_label_desc
from .loads import load_tree
from .trees import TensorTree, TensorLeaf, RegionLeaf, RegionTree, Leaf, Tree



class ImageCollection(dict):

    def __getitem__(self, key):
        if key not in self:
            self[key] = list()
        return super().__getitem__(key)

    def __add__(self, images):
        new_images = ImageCollection()
        for old_images in (self, images):
            for key in old_images.keys():
                new_images[key].extend(old_images[key])
        return new_images


class _FileInfo:
    def __init__(self, filepath):
        self.filepath = filepath
        self.dirname = os.path.dirname(filepath)
        self.basename = os.path.basename(filepath)
        self.ext = self._get_ext()
        self.parts = self._get_parts()
        self.suffix = self.parts[-1]

    def _get_ext(self):
        if self.basename.endswith('.nii.gz'):
            return '.nii.gz'
        else:
            return '.' + self.basename.split('.')[-1]

    def _get_parts(self):
        return self.basename.replace(self.ext, '').split('_')


class Loader:
    def __init__(self, files):
        self.files = files
        self.images = ImageCollection()

    def load(self):
        for f in self.files:
            if self._is_correct_type(f):
                self.images[f.name].append(self._create(f))

    def _create(self, f):
        raise NotImplementedError

    def _is_correct_type(self, f):
        raise NotImplementedError


class ImageLoader:

    def _create(self, f):
        return Image(filepath=f.filepath)

    def _is_correct_type(self, f):
        return f.suffix in Config().image_suffixes


class LabelLoader:

    def __init__(self, files, labels, pairs):
        super().__init__(files)
        self.labels = labels
        self.pairs = pairs

    def _create(self, f):
        return Label(filepath=f.filepath, labels=self.labels, pairs=self.pairs)

    def _is_correct_type(self, f):
        return f.suffix in Config().label_suffixes


class MaskLoader:

    def _create(self, f):
        return Mask(filepath=f.filepath, cropping_shape=Config().crop_shape)

    def _is_correct_type(self, f):
        return f.suffix in Config().mask_suffixes


class BoundingBoxLoader:

    def _create(self, f):
        return BoundingBox(filepath=f.filepath)

    def _is_correct_type(self, f):
        return f.suffix in Config().bbox_suffixes


# def __getitem__(self, key):
#     if isinstance(key, int):
#         key = list(self.images.keys())[key]
#     return self.images[key]


class Image:
    """Image

    Attributes:
        load_dtype (type): Data type of the internal storage
        output_dtype (type): Data type of the output
        filepath (str): The path to the file to load
        data (numpy.array): The image data
        on_the_fly (bool): If load the data on the fly
        message (list of str): The message for printing
        interp_order (int): The interpolation order of the image
        _data (numpy.array): Internal reference to the data; used for on the fly
         
    """
    load_dtype = np.float32
    output_dtype = np.float32

    def __init__(self, filepath=None, data=None, on_the_fly=True, message=[]):
        """Initialize

        Raises:
            RuntimeError: filepath and data are both None. The class should load
                from filepath or data
            RuntimeError: filepath is not None, data is None, and on_the_fly is
                True. If the class is initialized from data, on_the_fly should
                be False

        """
        if filepath is None and data is None:
            raise RuntimeError('"filepath" and "data" should not be both None')

        if data is not None and on_the_fly:
            error = '"on_the_fly" should be False if initialize from data'
            raise RuntimeError(error)

        self.filepath = filepath
        self.on_the_fly = on_the_fly
        self._data = data
        self.message = message
        self.interp_order = 1

    @property
    def data(self):
        """Get data

        Returns:
            result (numpy.array): The image data

        """
        if self.on_the_fly:
            return self._load()
        else:
            if self._data is None:
                self._data = self._load()
            return self._data

    def _load(self):
        data = load(self.filepath, self.load_dtype)
        if len(data.shape) == 3:
            data = data[None, ...]
        return data

    @property
    def output(self):
        """Get output

        self.output will be used by .datasets.Dataset to yield data. self.data
        will mainly be used by .workers.Worker to process

        Returns:
            result (numpy.array): The output of the image

        """
        return self.data.astype(self.output_dtype)

    def __str__(self):
        return ' '.join([os.path.basename(self.filepath)] + self.message)

    def update(self, data, message):
        """Create a new instance with data"""
        message =  self.message + [message]
        new_image = self.__class__(self.filepath, data, False, message)
        return new_image

    @property
    def shape(self):
        return self.data.shape # TODO


class Label(Image):
    """Label Image

    Attributes:
        labels (dict): The key is the label value and the dict value is the name
            of the label
        pairs (list): Each is a pair of left/right corresponding labels

    """
    load_dtype = np.uint8
    output_dtype = np.int64

    def __init__(self, filepath=None, data=None, on_the_fly=True, message=[],
                 labels=dict(), pairs=list()):
        super().__init__(filepath, data, on_the_fly, message)
        self.interp_order = 0
        self.labels = labels
        self.pairs = pairs

    def update(self, data, message, labels=None, pairs=None):
        labels = self.labels if labels is None else labels
        pairs = self.pairs if pairs is None else pairs
        message =  self.message + [message]
        new_image = self.__class__(self.filepath, data, False, message,
                                   labels, pairs)
        return new_image

    def normalize(self):
        """Convert label values into 0 : num_labels

        Returns:
            result (Label): The normalized label image

        """
        if len(self.labels) == 0:
            label_values = np.unique(self.data)
        else:
            label_values = sorted(self.labels.values())
        data = np.digitize(self.data, label_values, right=True)
        new_label_values = np.arange(len(label_values))
        mapping = {o: n for o, n in zip(label_values, new_label_values)}
        labels = {k: mapping[v] for k, v in self.labels.items()}
        pairs = [[mapping[p] for p in pair] for pair in self.pairs]
        result = self.update(data, 'label_norm', labels=labels, pairs=pairs)
        return result


class Mask(Image):
    """Mask

    Attributes:
        cropping_shape (list of int): The shape of the cropped

    """
    load_dtype = np.uint8
    output_dtype = np.int64

    def __init__(self, filepath=None, data=None, on_the_fly=True, message=[],
                 cropping_shape=[128, 96, 96]):
        super().__init__(filepath, data, on_the_fly, message)
        self.interp_order = 0
        self.cropping_shape = cropping_shape
        self._bbox = None
        
    def calc_bbox(self):
        """Calculate the bounding box

        """
        bbox = calc_bbox3d(self.data)
        self._bbox = resize_bbox3d(bbox, self.cropping_shape)

    def crop(self, image):
        """Crop another image

        Args:
            image (Image): The other image to crop

        Returns
            image (Image): The cropped image

        """
        if self._bbox is None:
            self.calc_bbox()
        cropped = crop3d(image.data, self._bbox)[0]
        message =  image.message + ['crop']
        #TODO
        if isinstance(image, HierachicalLabel):
            new_image = image.__class__(image.filepath, cropped, False, message,
                                        labels=image.labels, pairs=image.pairs,
                                        tree=image.region_tree)
        elif isinstance(image, Label):
            new_image = image.__class__(image.filepath, cropped, False, message,
                                        labels=image.labels, pairs=image.pairs)
        elif isinstance(image, Mask):
            new_image = image.__class__(image.filepath, cropped, False, message,
                                        cropping_shape=image.cropping_shape)
        else:
            new_image = image.__class__(image.filepath, cropped, False, message)
        return new_image

    @property
    def shape(self):
        return self.cropping_shape

    def update(self, data, message):
        message =  self.message + [message]
        new_image = self.__class__(self.filepath, data, False, message,
                                   cropping_shape=self.cropping_shape)
        return new_image


class BoundingBox(Image):
    """Bounding box of an image

    A binary image will be kept for .workers.Worker to process. The output is
    an array of start and stop along the x, y, and z axes
    
    TODO:
        support loading from .csv file

    """
    load_dtype = np.uint8
    output_dtype = np.float32

    def __init__(self, filepath=None, data=None, on_the_fly=True, message=[]):
        super().__init__(filepath, data, on_the_fly, message)
        self.interp_order = 0

    @property
    def output(self):
        bbox = calc_bbox3d(self.data)
        output = list()
        for b in bbox:
            output.extend((b.start, b.stop))
        return np.array(output, dtype=self.output_dtype)
