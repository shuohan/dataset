# -*- coding: utf-8 -*-

"""Implement Image to handle data

"""
import os
import numpy as np
from glob import glob
from enum import Enum, auto
from collections import defaultdict
from image_processing_3d import calc_bbox3d, resize_bbox3d, crop3d

from .config import Config
from .loads import load, load_label_desc, load_label_hierachy, Hierachy
from .tensor_tree import TensorTree, TensorLeaf


class ImageType(Enum):
    image = auto()
    label = auto()
    mask = auto()
    bounding_box = auto()
    hierachical_label = auto()


class ImageCollection(defaultdict):
    """Image collection

    Each key corresponds images of the same subject, the value is a list of
    Image instances

    """
    def __init__(self, *args, **kwargs):
        super().__init__(list, *args, **kwargs)

    def split(self, indicies):
        """Split images into two instances of ImageCollection

        Args:
            indicies (list of int): The indices in self for the first collection 

        Returns:
            collection1 (ImageCollection): The collection of images
                corresponding to the input arg `indicies`
            collection2 (ImageCollection): The collection of images
                corresponding to the rest

        """
        indicies2 = sorted(list(set(range(len(self))) - set(indicies)))
        keys = np.array(list(self.keys()))
        collection1 = self.__class__({k: self[k] for k in keys[indicies]})
        collection2 = self.__class__({k: self[k] for k in keys[indicies2]})
        return collection1, collection2

    def copy(self):
        return self.__class__(self)

    def __getitem__(self, key):
        """Support return by index"""
        if key not in self and isinstance(key, int):
            key = list(self.keys())[key]
        return super().__getitem__(key)

    def __add__(self, other):
        """Merge two image collections

        Args:
            other (ImageCollection): The other collection to merge

        Returns:
            result (ImageCollection): Merged collection

        """
        result = self.copy()
        result.update(other)
        return result


class ImageLoader:
    """Load images

    Call self.load and access loaded images via self.images

    Attributes:
        dirname (str): The directory to the files
        ext (str): The extension name of the files
        id (str): The identifier of the dataset
        images (ImageCollection): The loaded images        
        _files (list of dict): The list of dict with file information. 'name':
            the name of the file; 'suffix': the suffix of the filename;
            'filepath': the path to the file

    """
    def __init__(self, dirname, id='', ext='.nii.gz'):
        self.dirname = dirname
        self.id = id
        self.ext = ext
        self.images = ImageCollection()
        self._files = self._gather_files()

    def _gather_files(self):
        files = list()
        for filepath in sorted(glob(os.path.join(self.dirname, '*'+self.ext))):
            parts = os.path.basename(filepath).replace(self.ext, '').split('_')
            name = os.path.join(self.id, parts[0])
            files.append(dict(name=name, suffix=parts[-1], filepath=filepath))
        return files

    def load(self, *image_types):
        """Load images

        Args:
            image_type (str): The type of the images to load

        """
        config = Config()
        for type in image_types:
            if ImageType[type] is ImageType.image:
                self._load(config.image_suffixes, Image)
            elif ImageType[type] is ImageType.label:
                desc_filepath = os.path.join(self.dirname, config.label_desc)
                l, p = load_label_desc(desc_filepath)
                self._load(config.label_suffixes, Label, labels=l, pairs=p)
            elif ImageType[type] is ImageType.hierachical_label:
                desc_filepath = os.path.join(self.dirname, config.label_desc)
                h_filepath = os.path.join(self.dirname, config.label_hierachy)
                labels, pairs = load_label_desc(desc_filepath)
                hierachy = load_label_hierachy(h_filepath)
                self._load(config.hierachical_label_suffixes, HierachicalLabel,
                           labels=labels, pairs=pairs, hierachy=hierachy)
            elif ImageType[type] is ImageType.mask:
                self._load(config.mask_suffixes, Mask,
                           cropping_shape=config.crop_shape)
            elif ImageType[type] is ImageType.bounding_box:
                self._load(config.bbox_suffixes, BoundingBox)

    def _load(self, suffixes, image_class, **kwargs):
        for file in self._files:
            if file['suffix'] in suffixes:
                bbox = image_class(filepath=file['filepath'], **kwargs)
                self.images[file['name']].append(bbox)

    def __getitem__(self, key):
        if isinstance(key, int):
            key = list(self.images.keys())[key]
        return self.images[key]


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


class HierachicalLabel(Label):
    def __init__(self, filepath=None, data=None, on_the_fly=True, message=[],
                 labels=dict(), pairs=list(), hierachy=None):
        super().__init__(filepath, data, on_the_fly, message, labels, pairs)
        self.hierachy = hierachy

    #TODO
    def update(self, data, message, labels=None, pairs=None):
        labels = self.labels if labels is None else labels
        pairs = self.pairs if pairs is None else pairs
        message =  self.message + [message]
        new_image = self.__class__(self.filepath, data, False, message,
                                   labels, pairs, self.hierachy)
        return new_image

    def print_hierachy(self):
        self._print_hierachy(self.hierachy)

    def _print_hierachy(self, hierachy):
        values = [str(v) for v in self._get_values(hierachy)]
        string = hierachy._name_to_print + ' ' + ', '.join(values)
        string = string + (' (%d)' % len(values))
        print(string)
        if isinstance(hierachy, Hierachy):
            for region in hierachy.children:
                self._print_hierachy(region)

    def _get_values(self, hierachy):
        return [self.labels[r] for r in hierachy.regions]

    def get_tensor_tree(self):
        return self._get_tensor_tree(self.hierachy)

    def _get_tensor_tree(self, hierachy, level=0):
        name = hierachy.name
        if isinstance(hierachy, Hierachy):
            subtrees = list()
            for region in hierachy.children:
                subtrees.append(self._get_tensor_tree(region, level=level+1))
            data = self._get_masks(hierachy)
            tensor_leaf = TensorTree(name, data, subtrees, level=level)
        else:
            tensor_leaf = TensorLeaf(name, level=level)
        return tensor_leaf

    def _get_masks(self, hierachy):
        masks = list()
        for region in hierachy.children:
            values = self._get_values(region)
            masks.append(np.logical_or.reduce([self.data==v for v in values]))
        masks = np.vstack(masks).astype(self.output_dtype)
        return masks


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
                                        hierachy=image.hierachy)
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
