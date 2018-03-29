# -*- coding: utf-8 -*-

from .data import Data3d
from .data_decorators import Cropping3d, Interpolating3d, Flipping3d
from .transformers import Flipper, Rotator, Deformer


class Data3dFactory:

    def __init__(self, dim=1, label_pairs=[], max_angle=10, sigma=5, scale=8,
                 get_data_on_the_fly=False, transpose4d=True, types=['none']):

        self.dim = dim
        self.label_pairs = label_pairs
        self.max_angle = max_angle
        self.sigma = sigma
        self.scale = scale
        self.get_data_on_the_fly = get_data_on_the_fly
        self.transpose4d = transpose4d
        self.types = types

        self.data = dict()

    def create_data(self, *filepaths):
        self.data = dict() 
        if 'none' in self.types:
            self._create_none(filepaths)
            if 'rotation' in self.types:
                self._create_rotated()
            if 'deformation' in self.types:
                self._create_deformed()
            if 'flipping' in self.types:
                self._create_flipped()
                if 'rotation' in self.types:
                    self._create_rotated_flipped()
                if 'deformation' in self.types:
                    self._create_deformed_flipped()

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


class Data3dFactoryDecorator(Data3dFactory):

    def __init__(self, data3d_factory):
        self.factory = data3d_factory
        self.types = self.factory.types

    def create_data(self, *filepaths):
        self.factory.create_data(*filepaths[:-1])
        super().create_data(*filepaths)


class Data3dFactoryCropper(Data3dFactoryDecorator):

    def __init__(self, data3d_factory, cropping_shape):
        super().__init__(data3d_factory)
        self.uncropped_data = dict()
        self.cropping_shape = cropping_shape

    def create_data(self, *filepaths):
        self.uncropped_data = dict()
        super().create_data(*filepaths)

    def _create_none(self, filepaths):
        mask = Data3d(filepaths[-1], self.factory.get_data_on_the_fly,
                      self.factory.transpose4d)
        self._crop('none', mask)

    def _create_flipped(self):
        mask = self._transform('none', 'flipped', Flipping3d)
        self._crop('flipped', mask)

    def _create_rotated(self):
        mask = self._transform('none', 'rotated', Interpolating3d)
        self._crop('rotated', mask)

    def _create_rotated_flipped(self):
        mask = self._transform('flipped', 'rotated_flipped', Interpolating3d)
        self._crop('rotated_flipped', mask)

    def _create_deformed(self):
        mask = self._transform('none', 'deformed', Interpolating3d)
        self._crop('deformed', mask)

    def _create_deformed_flipped(self):
        mask = self._transform('flipped', 'deformed_flipped', Interpolating3d)
        self._crop('deformed_flipped', mask)

    def _transform(self, source_key, target_key, Transforming):
        data = self.factory.data[target_key][0]
        mask = Transforming(self.uncropped_data[source_key][-1],
                            data.transformer,
                            get_data_on_the_fly=data.get_data_on_the_fly)
        return mask

    def _crop(self, key, mask):
        self.uncropped_data[key] = (*self.factory.data[key], mask)
        self.data[key] = tuple([Cropping3d(d, mask, self.cropping_shape,
                                           d.get_data_on_the_fly)
                                for d in self.factory.data[key]])
