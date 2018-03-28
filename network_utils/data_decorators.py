# -*- coding: utf-8 -*-

from image_processing_3d import crop3d, calc_bbox3d, resize_bbox3d

from .data import Data3d

class DataDecorator:

    def __init__(self, data):
        self.data = data

    def get_data(self):
        return self.data.get_data()


class Cropping3d(DataDecorator):

    def __init__(self, data3d, cropping_shape, load_on_the_fly=True):
        super().__init__(data3d)
        self.cropping_shape = cropping_shape
        self.load_on_the_fly = load_on_the_fly
        mask_filepath = self.data.filepath.replace('image', 'mask')
        self.mask = Data3d(mask_filepath, self.data.load_on_the_fly,
                           self.data.transpose4d)

        self._source_bbox = None
        self._target_bbox = None
        self._cropped_data = None

    def get_data(self):
        if self.load_on_the_fly:
            cropped_data, self._source_bbox, self._target_bbox = self._crop()
            return cropped_data
        else:
            if self._cropped_data is None:
                self._cropped_data self._source_bbox, self._target_bbox = \
                        self._crop()
            return self._cropped_data

    def _crop(self):
        data = self.data.get_data()
        mask = self.mask.get_data()
        bbox = calc_bbox3d(mask)
        resized_bbox = resize_bbox3d(bbox, self.cropping_shape)
        cropped, source_bbox, target_bbox = crop3d(data, resized_bbox)
        return cropped, source_bbox, target_bbox


class Binarizing3d(Dataset3dDecorator):

    def __init__(self, data, binarizer, load_on_the_fly=True):
        super().__init__(self, data)
        self.binarizer = binarizer
        self.load_on_the_fly = load_on_the_fly

        self._binarized_data = None

    def get_data(self):
        if self.load_on_the_fly:
            return self._binarize()
        else:
            if self._binarized_data is None:
                self._binarized_data = self._binarize()
            return self._binarized_data

    def _binarize(self):
        data = self.data.get_data()
        if not hasattr(self.binarizer, 'classes_'):
            self.binarizer.fit(np.unique(data))
        result = self.binarizer.transform(data)
        result = np.rollaxis(result, -1) # channels first
        return result
