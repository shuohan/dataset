# -*- coding: utf-8 -*-

"""Implement Image to handle data

"""
import os
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from image_processing_3d import calc_bbox3d, resize_bbox3d, crop3d

from .config import Config


IMAGE_EXT = '.nii*'


class FileSearcher:

    def __init__(self, dirname):
        self.dirname = dirname
        self.files = None
        self.label_file = None

    def search(self):
        filepaths = sorted(Path(self.dirname).glob('*' + IMAGE_EXT))
        self.files = [FileInfo(fp) for fp in filepaths]
        self.label_file = os.path.join(self.dirname, Config().label_desc)
        return self


class FileInfo:
    def __init__(self, filepath=''):
        self.filepath = filepath
        self.dirname = os.path.dirname(self.filepath)
        self.basename = os.path.basename(self.filepath)
        self.ext = self._get_ext()
        self._parts = self._get_parts()
        self.name = self._parts[0] if self._parts else ''
        self.suffix = self._parts[-1] if self._parts else ''

    def _get_ext(self):
        if self.basename.endswith('.nii.gz'):
            return '.nii.gz'
        else:
            return '.' + self.basename.split('.')[-1]

    def _get_parts(self):
        return self.basename.replace(self.ext, '').split('_')

    def __str__(self):
        fields = ['filepath', 'dirname', 'basename', 'ext', 'name', 'suffix']
        str_len = max([len(f) for f in fields])
        message = list()
        for field in fields:
            pattern = '%%%ds: %%s' % str_len
            message.append(pattern % (field, getattr(self, field)))
        return '\n'.join(message)


class LabelMapping(dict):
    def __setitem__(self, key, val):
        raise RuntimeError('class Labels does not support changing contents')
    def __getattr__(self, key):
        if key not in self.__dict__:
            return self[key]


class LabelInfo:

    def __init__(self, filepath=None, labels=None, pairs=None):
        self.filepath = filepath
        if self.filepath is None:
            if labels is None or pairs is None:
                message = ('"labels" or "pairs" cannot be both None when '
                           '"filepath" is None.')
                raise RuntimeError(message)
            else:
                self.labels = labels
                self.pairs = pairs
        else:
            self.labels = self._load_labels()
            self.pairs = self._load_pairs()

    def _load_labels(self):
        labels = self._load_json()['labels']
        return LabelMapping(**labels)

    def _load_pairs(self):
        pairs = self._load_json()['pairs']
        return tuple(tuple(p) for p in pairs)

    def _load_json(self):
        with open(self.filepath) as jfile:
            contents = json.load(jfile)
        return contents

    def __hash__(self):
        labels = tuple(self.labels.keys()) + tuple(self.labels.values())
        return hash(labels + self.pairs)

    def __eq__(self, label_info):
        return hash(self) == hash(label_info)


class ImageCollection(dict):

    def __getitem__(self, key):
        if key not in self:
            self[key] = list()
        return super().__getitem__(key)

    def append(self, image):
        self[image.info.name].append(image)

    def __add__(self, images):
        new_images = ImageCollection()
        for old_images in (self, images):
            for key in old_images.keys():
                new_images[key].extend(old_images[key])
        return new_images

    def __radd__(self, images):
        return self.__add__(images)


class Loader:
    def __init__(self, file_searcher):
        self.file_searcher = file_searcher
        self.images = ImageCollection()

    def load(self):
        for f in self.file_searcher.files:
            if self._is_correct_type(f):
                self.images.append(self._create(f))
        return self

    def _create(self, f):
        raise NotImplementedError

    def _is_correct_type(self, f):
        raise NotImplementedError


class ImageLoader(Loader):

    def _create(self, f):
        return Image(info=f)

    def _is_correct_type(self, f):
        return f.suffix in Config().image_suffixes


class LabelLoader(Loader):

    def __init__(self, file_searcher):
        super().__init__(file_searcher)

    def _create(self, f):
        label_info = LabelInfo(self.file_searcher.label_file)
        return Label(info=f, label_info=label_info)

    def _is_correct_type(self, f):
        return f.suffix in Config().label_suffixes


class MaskLoader(Loader):

    def _create(self, f):
        return Mask(info=f, cropping_shape=Config().crop_shape)

    def _is_correct_type(self, f):
        return f.suffix in Config().mask_suffixes


class BoundingBoxLoader(Loader):

    def _create(self, f):
        return BoundingBox(info=f)

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

    def __init__(self, info=None, data=None, on_the_fly=True, message=[]):
        """Initialize

        Raises:
            RuntimeError: filepath and data are both None. The class should load
                from filepath or data
            RuntimeError: filepath is not None, data is None, and on_the_fly is
                True. If the class is initialized from data, on_the_fly should
                be False

        """
        if info is None and data is None:
            raise RuntimeError('"info" and "data" should not be both None')

        if data is not None and on_the_fly:
            error = '"on_the_fly" should be False if initialize from data'
            raise RuntimeError(error)

        self.info = info
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
        data = load(self.info.filepath, self.load_dtype)
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
        message = ['%s %s:' % (self.info.name, self.info.suffix)] + self.message
        return ' '.join(message)

    def update(self, data, message):
        """Create a new instance with data"""
        message =  self.message + [message]
        new_image = self.__class__(self.info, data, False, message)
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

    def __init__(self, info=None, data=None, on_the_fly=True, message=[],
                 label_info=None):
        super().__init__(info, data, on_the_fly, message)
        self.interp_order = 0
        self.labels = labels
        self.pairs = pairs

    def update(self, data, message, labels=None, pairs=None):
        labels = self.labels if labels is None else labels
        pairs = self.pairs if pairs is None else pairs
        message =  self.message + [message]
        new_image = self.__class__(self.info, data, False, message,
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

    @property
    def normalized_label_info(self):
        pass

    def _get_label_values(self):
        pass

    def _get_default_label_info(self):
        label_values =  np.unique(self.data)


class Mask(Image):
    """Mask

    Attributes:
        cropping_shape (list of int): The shape of the cropped

    """
    load_dtype = np.uint8
    output_dtype = np.int64

    def __init__(self, info=None, data=None, on_the_fly=True, message=[],
                 cropping_shape=[128, 96, 96]):
        super().__init__(info, data, on_the_fly, message)
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
            new_image = image.__class__(image.info, cropped, False, message,
                                        labels=image.labels, pairs=image.pairs,
                                        tree=image.region_tree)
        elif isinstance(image, Label):
            new_image = image.__class__(image.info, cropped, False, message,
                                        labels=image.labels, pairs=image.pairs)
        elif isinstance(image, Mask):
            new_image = image.__class__(image.info, cropped, False, message,
                                        cropping_shape=image.cropping_shape)
        else:
            new_image = image.__class__(image.info, cropped, False, message)
        return new_image

    @property
    def shape(self):
        return self.cropping_shape

    def update(self, data, message):
        message =  self.message + [message]
        new_image = self.__class__(self.info, data, False, message,
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

    def __init__(self, info=None, data=None, on_the_fly=True, message=[]):
        super().__init__(info, data, on_the_fly, message)
        self.interp_order = 0

    @property
    def output(self):
        bbox = calc_bbox3d(self.data)
        output = list()
        for b in bbox:
            output.extend((b.start, b.stop))
        return np.array(output, dtype=self.output_dtype)
