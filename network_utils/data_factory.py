# -*- coding: utf-8 -*-

from .data import Data3d
from .data_decorators import Cropping3d, Interpolating3d, Flipping3d
from .transformers import Flipper, Rotator, Deformer


class Data3dFactory:

    def __init__(self, dim=1, label_pairs=[], max_angle=10, sigma=5, scale=8,
                 get_data_on_the_fly=False, transpose4d=True):

        self.dim = dim
        self.label_pairs = label_pairs
        self.max_angle = max_angle
        self.sigma = sigma
        self.scale = scale
        self.get_data_on_the_fly = get_data_on_the_fly
        self.transpose4d = transpose4d

        self.data = dict()

    def create_data(self, *filepaths, types=['none']):
        self.data = dict() 
        if 'none' in types:
            self._create_none(filepaths)
            if 'rotation' in types:
                self._create_rotated()
            if 'deformation' in types:
                self._create_deformed()
            if 'flipping' in types:
                self._create_flipped()
                if 'rotation' in types:
                    self._create_rotated_flipped()
                if 'deformation' in types:
                    self._create_deformed_flipped()
        return self.data

    def _create_one(self, filepaths):
        raise NotImplementedError

    def _create_flipped(self):
        raise NotImplementedError

    def _create_rotated(self):
        raise NotImplementedError

    def _create_rotated_flipped(self):
        raise NotImplementedError

    def _create_deformed(self):
        raise NotImplementedError

    def _create_deformed_flipped(self):
        raise NotImplementedError


class TrainingDataFactory(Data3dFactory):

    def _create_none(self, filepaths):
        image = Data3d(filepaths[0], self.get_data_on_the_fly, self.transpose4d)
        label = Data3d(filepaths[1], self.get_data_on_the_fly, self.transpose4d)
        self.data['none'] = (image, label)

    def _create_flipped(self):
        flipper = Flipper(dim=self.dim)
        image = Flipping3d(self.data['none'][0], flipper, label_pairs=[],
                           get_data_on_the_fly=self.get_data_on_the_fly)
        label = Flipping3d(self.data['none'][1], flipper,
                           label_pairs=self.label_pairs,
                           get_data_on_the_fly=self.get_data_on_the_fly)
        self.data['flipped'] = (image, label)
    
    def _create_rotated(self):
        self.data['rotated'] = self._rotate(self.data['none'])

    def _create_rotated_flipped(self):
        self.data['rotated_flipped'] = self._rotate(self.data['flipped'])

    def _rotate(self, data):
        rotator = Rotator(max_angle=self.max_angle)
        image = Interpolating3d(data[0], rotator, order=1,
                                get_data_on_the_fly=True)
        label = Interpolating3d(data[1], rotator, order=0,
                                get_data_on_the_fly=True)
        return image, label

    def _create_deformed(self):
        self.data['deformed'] = self._deform(self.data['none'])

    def _create_deformed_flipped(self):
        self.data['deformed_flipped'] = self._deform(self.data['flipped'])

    def _deform(self, data):
        shape = data[0].get_data().shape[-3:]
        deformer = Deformer(shape, self.sigma, self.scale)
        image = Interpolating3d(data[0], deformer, order=1,
                                get_data_on_the_fly=True)
        label = Interpolating3d(data[1], deformer, order=0,
                                get_data_on_the_fly=True)
        return image, label


class DecoratedData3dFactory(Data3dFactory):

    def __init__(self, data3d_factory):
        self.factory = data3d_factory


class CroppedData3dFactory(DecoratedData3dFactory):

    def __init__(self, data3d_factory, cropping_shape):
        super().__init__(data3d_factory)
        self.cropping_shape = cropping_shape

    def create_data(self, *filepaths, types=['none']):
        data = self.factory.create_data(*filepaths[:-1], types=types)
        super().create_data(*filepaths, types=types)
        for key in self.data.keys():
            cropped = [Cropping3d(d, self.data[key][-1], self.cropping_shape,
                                  d.get_data_on_the_fly)
                       for d in self.data[key][:-1]]
            self.data[key] = cropped

        return self.data

    def _create_none(self, filepaths):
        mask = Data3d(filepaths[-1], self.factory.get_data_on_the_fly,
                      self.factory.transpose4d)
        self.data['none'] = (*self.factory.data['none'], mask)

    def _create_flipped(self):
        flipper = self.factory.data['flipped'][0].transformer
        get_data_on_the_fly = self.factory.get_data_on_the_fly
        mask = Flipping3d(self.data['none'][-1], flipper, label_pairs=[],
                          get_data_on_the_fly=get_data_on_the_fly)
        self.data['flipped'] = (*self.factory.data['flipped'], mask)

    def _create_rotated(self):
        rotator = self.factory.data['rotated'][0].transformer
        mask = Interpolating3d(self.data['none'][-1], rotator, True)
        self.data['rotated'] = (*self.factory.data['rotated'], mask)

    def _create_rotated_flipped(self):
        rotator = self.factory.data['rotated_flipped'][0].transformer
        mask = Interpolating3d(self.data['flipped'][-1], rotator, True)
        self.data['rotated_flipped'] = (*self.factory.data['rotated_flipped'],
                                        mask)

    def _create_deformed(self):
        shape = self.data['none'][-1].get_data().shape[-3:]
        deformer = self.factory.data['deformed'][0].transformer
        mask = Interpolating3d(self.data['none'][-1], deformer, True)
        self.data['deformed'] = (*self.factory.data['deformed'], mask)

    def _create_deformed_flipped(self):
        shape = self.data['flipped'][-1].get_data().shape[-3:]
        deformer = self.factory.data['deformed_flipped'][0].transformer
        mask = Interpolating3d(self.data['flipped'][-1], deformer, True)
        self.data['deformed_flipped'] = (*self.factory.data['deformed_flipped'],
                                         mask)
