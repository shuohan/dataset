# -*- coding: utf-8 -*-

"""Implement Image to handle data

"""
import os
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
        if self.on_the_fly:
            return load(self.filepath)
        else:
            if self._data is None:
                self._data = load(self.filepath)
            return self._data

    def __str__(self):
        return ' '.join([os.path.basename(self.filepath)] + self.message)

    def update(self, data, message):
        """Create a new instance with data"""
        message =  self.message + [message]
        return self.__class__(self.filepath, data, False, message)

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
        print(new_image.shape)
        return new_image

    @property
    def shape(self):
        return self.cropping_shape
