# -*- coding: utf-8 -*-

"""Images handling the data.

"""
import os
import json
import numpy as np
from pathlib import Path
from image_processing_3d import calc_bbox3d, resize_bbox3d, crop3d

from .config import Config


IMAGE_EXT = '.nii*'


def load(filepath, dtype):
    """Loads an image from hard drive.

    Args:
        filepath (str): The path to the file to load.
        dtype (type): The data type of the loaded data.

    Returns:
        numpy.ndarray: Loaded data.
    
    """
    if filepath.endswith('.npy'):
        return _load_npy(filepath, dtype)
    elif filepath.endswith('.nii') or filepath.endswith('.nii.gz'):
        return _load_nii(filepath, dtype)


def _load_npy(filepath, dtype):
    """Loads a ``".npy"`` file."""
    return np.load(filepath).astype(dtype)


def _load_nii(filepath, dtype):
    """Load a ``".nii"`` or ``".nii.gz"`` file."""
    import nibabel as nib
    return nib.load(filepath).get_data().astype(dtype)


class FileSearcher:
    """Searches files in a directory.

    Call the method :meth:`search` to search the files, then use the attributes
    :attr:`files` and :attr:`label_file` to get the resutls.

    Attributes:
        dirname (str): The name of the directory to search.
        files (list[FileInfo]): The searched files.
        label_file (str): The file specifies the delineation labels.

    """
    def __init__(self, dirname):
        self.dirname = dirname
        self.files = None
        self.label_file = None

    def search(self):
        """Searches the files.

        Use the attributes :attr:`files` and :attr:`label_file` to access the
        searching results.

        Returns:
            FileSearcher: The instance itself.

        """
        filepaths = sorted(Path(self.dirname).glob('*' + IMAGE_EXT))
        self.files = [FileInfo(fp) for fp in filepaths]
        self.label_file = os.path.join(self.dirname, Config.label_desc)
        return self


class FileInfo:
    """Stores the information of a file.

    The image file should be Nifti (``".nii"`` or ``".nii.gz"``). The basename
    contains the subject identifier (the attribute :attr:`name`) and image type
    (the attribute :attr:`suffix`) separated by ``"_"``, for example,
    ``"subj1_image.nii.gz"``.

    Attributes:
        filepath (str): The path to the file.
        dirname (str): The directory of the file.
        basename (str): The basename of the file.
        ext (str): The extension of the file.
        name (str): The name identifying the file. :class:`FileInfo` of the same
            subject should be the same.
        suffix (str): The suffix specifying the image type such as ``mask``,
            ``image``, etc.

    """
    def __init__(self, filepath=''):
        self.filepath = str(filepath)
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
    """Maps label name to label value.

    This class is an immutable dict. It also supports accessing with a key via
    ``.`` like a class attribute if the key is in it. Initialize an instance as:

    >>> LabelMapping(**dict_contents)
    
    """
    def __setitem__(self, key, val):
        raise RuntimeError('class Labels does not support changing contents')
    def __getattr__(self, key):
        if key not in self.__dict__:
            return self[key]


class LabelInfo:
    """Handles label information.

    This class can be compared. Two instances of this class equal each other
    when the attributes :attr:`labels` and :attr:`pairs` are the same,
    respectively. This class can also be used as a :class:`dict` key or in
    :class:`set`.

    This class can be iniatilzed by only the attribute :attr:`filepath` or the
    already loaded variables ``labels`` and ``pairs``. Since the class performs
    type conversion, ``labels`` can be :class:`dict` and ``pairs`` can be any
    iterable of iterable of :class:`int`.

    Attributes:
        filepath (str): The path to the label description .json file.
        labels (LabelMapping): The label name and value mapping.
        pairs (tuple[tuple[int]]): The pair of left and right labels. For
            example, the brain has left and right temporal lobes.

    Raises:
        RuntimeError: the input ``filepath`` is ``None`` and any of ``labels``
            and ``pairs`` is ``None``.

    """
    def __init__(self, filepath=None, labels=None, pairs=None):
        self.filepath = filepath
        if self.filepath is None:
            if labels is None or pairs is None:
                message = ('"labels" or "pairs" cannot be both None when '
                           '"filepath" is None.')
                raise RuntimeError(message)
            else:
                self.labels = self._convert_labels(labels)
                self.pairs = self._convert_pairs(pairs)
        else:
            self.labels = self._load_labels()
            self.pairs = self._load_pairs()

    def _load_labels(self):
        labels = self._load_json()['labels']
        return self._convert_labels(labels)

    def _convert_labels(self, labels):
        return LabelMapping(**labels)

    def _load_pairs(self):
        pairs = self._load_json()['pairs']
        return self._convert_pairs(pairs)

    def _convert_pairs(self, pairs):
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

    def __str__(self):
        message = ['Labels:']
        str_len = max([len(key) for key in self.labels.keys()])
        for key, value in self.labels.items():
            message.append(('%%%ds: %%s' % str_len) % (key, value))
        message.append('Pairs:')
        for pair in self.pairs:
            message.append('%d, %d' % pair)
        return '\n'.join(message)


class ImageCollection(dict):
    """Holds a collection of :class:`Image`.

    This class inherits from :class:`dict`. However, a key can only be
    :class:`str`. Each value is a :class:`list` of class :class:`Image`
    belonging to the same subject. The images can also be accsessed by an
    integer index via the method :meth:`at`. Adding two instances of this class
    will concatenate the images of the same key and join the the keys of the
    both instances.

    """
    def __getitem__(self, key):
        if type(key) is not str:
            raise RuntimeError('The key can only be str.')
        if key not in self:
            self[key] = list()
        return super().__getitem__(key)

    def at(self, index):
        """Returns an image group at an integer index."""
        if type(index) is not int:
            raise RuntimeError('The index can only be int')
        return list(self.values())[index]

    def append(self, image):
        """Appends an image to the key ``image.info.name``."""
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
    """Abstract class loading :class:`Image`.

    Rewrite methods :meth:`create` and :meth:`is_correct_type` to load different
    child classes of :class:`Image`.

    Attributes:
        file_searcher (FileSearcher): Searches the files.
        images (ImageCollection): The loaded images.

    """
    def __init__(self, file_searcher):
        self.file_searcher = file_searcher
        self.images = ImageCollection()

    def load(self):
        """Loads images found by the attribute :attr:`file_searcher`.
        
        Use the attribute :attr:`images` to access the loaded images after
        calling this method.
        
        Returns:
            Loader: The instance itself.

        """
        for f in self.file_searcher.files:
            if self.is_correct_type(f):
                self.images.append(self.create(f))
        return self

    def create(self, f):
        """Implements how to create an instance of class :class:`Image`.

        Args:
            f (FileInfo): The file to load.
        
        """
        raise NotImplementedError

    def is_correct_type(self, f):
        """Implements how to determine the correct image type.
        
        Args:
            f (FileInfo): The file to load.
        
        """
        raise NotImplementedError


class ImageLoader(Loader):
    """Loads :class:`Image`."""
    def create(self, f):
        return Image(info=f)
    def is_correct_type(self, f):
        return f.suffix in Config.image_suffixes


class LabelLoader(Loader):
    """Loads :class:`Label`."""
    def create(self, f):
        label_info = LabelInfo(self.file_searcher.label_file)
        return Label(info=f, label_info=label_info)
    def is_correct_type(self, f):
        return f.suffix in Config.label_suffixes


class MaskLoader(Loader):
    """Loads :class:`Mask`."""
    def create(self, f):
        return Mask(info=f, cropping_shape=Config.crop_shape)
    def is_correct_type(self, f):
        return f.suffix in Config.mask_suffixes


class BoundingBoxLoader(Loader):
    """Loads :class:`BoundingBox`."""
    def create(self, f):
        return BoundingBox(info=f)
    def is_correct_type(self, f):
        return f.suffix in Config.bbox_suffixes


class Image:
    """Handles an image.

    Attributes:
        load_dtype (type): Data type of the internal storage.
        output_dtype (type): Data type of the output.
        info (FileInfo): The file information.
        on_the_fly (bool): If to load the data on the fly.
        message (list[str]): The message for printing.
        interp_order (int): The interpolation order of the image.
         
    """
    load_dtype = np.float32
    output_dtype = np.float32

    def __init__(self, info=None, data=None, on_the_fly=True, message=[]):
        """Initialize

        Raises:
            RuntimeError: ``info`` and ``data`` are both ``None``. The class
                should load from either ``filepath`` or ``data``.
            RuntimeError: ``filepath`` is not ``None``, ``data`` is ``None``,
                and ``on_the_fly`` is ``True``. If the class is initialized from
                ``data``, ``on_the_fly`` should be ``False``.

        """
        if info is None and data is None:
            raise RuntimeError('"info" and "data" should not be both None.')
        if data is not None and on_the_fly:
            error = '"on_the_fly" should be False if initialize from data.'
            raise RuntimeError(error)

        self.info = info
        self.on_the_fly = on_the_fly
        self._data = data
        self.message = message
        self.interp_order = 1

    @property
    def data(self):
        """Returns data in :class:`numpy.ndarray`."""
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
        """Returns output.

        The output is derived from the attribute :attr:`data`. This is mainly
        used by the class :class:`Dataset` to yield data since the internally
        stored data can be different from the desired output.

        Returns:
            numpy.ndarray: The output of the image.

        """
        return self.data.astype(self.output_dtype)

    def __str__(self):
        message = ['%s %s:' % (self.info.name, self.info.suffix)] + self.message
        return ' '.join(message)

    def update(self, data, message):
        """Creates a new instance from ``data`` and ``message``."""
        message =  self.message + [message]
        new_image = self.__class__(self.info, data, False, message)
        return new_image

    @property
    def shape(self):
        """Returns the shape of the data in :class:`tuple`."""
        return self.data.shape # TODO


class Label(Image):
    """Handles a label image.

    This class can normalize itself so the label values will become a series of
    consecutive numbers.

    Attributes:
        label_info (LabelInfo): The label description.

    """
    load_dtype = np.uint8
    output_dtype = np.int64

    def __init__(self, info=None, data=None, on_the_fly=True, message=[],
                 label_info=None):
        super().__init__(info, data, on_the_fly, message)
        self.interp_order = 0
        if not isinstance(label_info, LabelInfo):
            label_info = self._get_default_label_info()
        self.label_info = label_info

    def _get_default_label_info(self):
        label_values = np.unique(self.data)
        labels = {str(l): l  for l in label_values}
        pairs = tuple()
        label_info = LabelInfo(labels=labels, pairs=pairs)
        return label_info

    @property
    def normalized_label_info(self):
        """Returns the label info the the normalized label image."""
        label_values = self._get_label_values()
        norm_label_values = np.arange(len(label_values))
        mapping = {o: n for o, n in zip(label_values, norm_label_values)}
        labels = {k: mapping[v] for k, v in self.label_info.labels.items()}
        pairs = [[mapping[p] for p in pair] for pair in self.label_info.pairs]
        norm_label_info = LabelInfo(labels=labels, pairs=pairs)
        return norm_label_info

    def update(self, data, message, label_info=None):
        label_info = self.label_info if label_info is None else label_info
        message =  self.message + [message]
        new_image = self.__class__(self.info, data, False, message, label_info)
        return new_image

    def normalize(self):
        """Normalizes the label image.
        
        Uses 0 to (the number of labels - 1) as the labels and reset the label
        values in the ascent order. The atrribute :attr:`label_info` will also
        be updated accordingly.

        Returns:
            Label: The normalized label image.
        
        """
        label_values = self._get_label_values()
        data = np.digitize(self.data, label_values, right=True)
        label_info = self.normalized_label_info
        result = self.update(data, 'norm_label', label_info=label_info)
        return result

    def _get_label_values(self):
        return sorted(self.label_info.labels.values())


class Mask(Image):
    """Handles a mask image.

    This class can crop other instances of :class:`Image`. Call the method
    :meth:`calc_bbox` to calculate the bounding box for cropping then access the
    bounding box by the attribute :attr:`bbox`. However, the method :meth:`crop`
    calls :meth:`calc_bbox` so it does not need to be called separately.

    Attributes:
        cropping_shape (list[int]): The shape of the cropped.
        bbox (tuple[slice]): The bounding box specifying the starts and stops
            along the x, y, and z axes around the mask.

    """
    load_dtype = np.uint8
    output_dtype = np.int64

    def __init__(self, info=None, data=None, on_the_fly=True, message=[],
                 cropping_shape=[128, 96, 96]):
        super().__init__(info, data, on_the_fly, message)
        self.interp_order = 0
        self.cropping_shape = cropping_shape
        self.bbox = None
        
    def calc_bbox(self):
        """Calculates the bounding box."""
        bbox = calc_bbox3d(self.data)
        self.bbox = resize_bbox3d(bbox, self.cropping_shape)

    def crop(self, image):
        """Crops another image using this mask.

        Args:
            image (Image): The other image to crop.

        Returns
            Image: The cropped image.

        """
        if self.bbox is None:
            self.calc_bbox()
        cropped = crop3d(image.data, self.bbox)[0]
        new_image = image.update(cropped, 'crop')
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
    """Handles a bounding box of an image.

    The bounding box is internally stored as a binary mask, and all
    transformation will be applied to the mask image. The mask will be converted
    to a bounding box described by corner coordinates only when calling the
    property :attr:`output`.

    """
    load_dtype = np.uint8
    output_dtype = np.float32

    def __init__(self, info=None, data=None, on_the_fly=True, message=[]):
        super().__init__(info, data, on_the_fly, message)
        self.interp_order = 0

    @property
    def output(self):
        """Returns an array of starts and stops along the x, y, and z axes."""
        bbox = calc_bbox3d(self.data)
        output = list()
        for b in bbox:
            output.extend((b.start, b.stop))
        return np.array(output, dtype=self.output_dtype)
