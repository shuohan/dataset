# -*- coding: utf-8 -*-

import numpy as np
from image_processing_3d import calc_bbox3d

from .data_factories import Data3dFactoryDecorator
from .data_decorators import DataDecorator


class BboxFactoryDecorator(Data3dFactoryDecorator):

    def _create_none(self, filepaths):
        result = self._calc_bbox(self.factory.data['none'])
        self.data['none'] = result

    def _create_flipped(self):
        result = self._calc_bbox(self.factory.data['flipped'])
        self.data['flipped'] = result

    def _create_rotated(self):
        result = self._calc_bbox(self.factory.data['rotated'])
        self.data['rotated'] = result

    def _create_deformed(self):
        result = self._calc_bbox(self.factory.data['deformed'])
        self.data['deformed'] = result

    def _create_translated(self):
        result = self._calc_bbox(self.factory.data['translated'])
        self.data['translated'] = result

    def _create_translated_flipped(self):
        result = self._calc_bbox(self.factory.data['translated_flipped'])
        self.data['translated_flipped'] = result

    def _create_rotated_flipped(self):
        result = self._calc_bbox(self.factory.data['rotated_flipped'])
        self.data['rotated_flipped'] = result

    def _create_deformed_flipped(self):
        result = self._calc_bbox(self.factory.data['deformed_flipped'])
        self.data['deformed_flipped'] = result

    def _calc_bbox(self, data):
        image = data[0]
        bbox = Bbox(data[1], get_data_on_the_fly=data[1].get_data_on_the_fly)
        return image, bbox


class Bbox(DataDecorator):

    def _get_data(self):
        bbox_slices = calc_bbox3d(self.data.get_data()[0, ...])
        bbox = list()
        for s in bbox_slices:
            bbox.append(s.start)
            bbox.append(s.stop)
        return np.array(bbox)
