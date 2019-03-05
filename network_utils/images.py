# -*- coding: utf-8 -*-

"""Implement Image to handle data

"""
import os
import numpy as np
from image_processing_3d import calc_bbox3d, resize_bbox3d, crop3d

from .loads import load


class Image:

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
            return load(self.filepath)
        else:
            if self._data is None:
                self._data = load(self.filepath)
            return self._data

    @property
    def output(self):
        """Get output

        self.output will be used by .datasets.Dataset to yield data. self.data
        will mainly be used by .workers.Worker to process

        Returns:
            result (numpy.array): The output of the image

        """
        return self.data

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

    def __init__(self, filepath=None, data=None, on_the_fly=True, message=[],
                 labels=[], pairs=[]):
        super().__init__(filepath, data, on_the_fly, message)
        self.interp_order = 0
        self.labels = labels
        self.pairs = pairs

    def update(self, data, message):
        message =  self.message + [message]
        new_image = self.__class__(self.filepath, data, False, message,
                                   self.labels, self.pairs)
        return new_image

    def binarize(self):
        # message =  self.message + ['binarize']
        # data = binarize(self.data, self.labels)
        # return self.__class__(self.filepath, data, False, message, self.labels,
        #                       self.label_pairs)
        return self


class Mask(Image):
    def __init__(self, filepath=None, data=None, on_the_fly=True, message=[],
                 cropping_shape=(128, 96, 96)):
        super().__init__(filepath, data, on_the_fly, message)
        self.interp_order = 0
        self.cropping_shape = cropping_shape
        self._bbox = None
        
    def calc_bbox(self):
        bbox = calc_bbox3d(self.data)
        self._bbox = resize_bbox3d(bbox, self.cropping_shape)

    def crop(self, image):
        if self._bbox is None:
            self.calc_bbox()
        cropped = crop3d(image.data, self._bbox)[0]
        message =  image.message + ['crop']
        new_image = image.__class__(image.filepath, cropped, False, message)
        return new_image

    @property
    def shape(self):
        return self.cropping_shape


class BoundingBox(Image):
    """Bounding box of an image

    A binary image will be kept for .workers.Worker to process. The output is
    an array of start and stop along the x, y, and z axes
    
    TODO:
        support loading from .csv file

    """
    def __init__(self, filepath=None, data=None, on_the_fly=True, message=[]):
        super().__init__(filepath, data, on_the_fly, message)
        self.interp_order = 0

    @property
    def output(self):
        bbox = calc_bbox3d(self.data)
        output = list()
        for b in bbox:
            output.extend((b.start, b.stop))
        return np.array(output)
