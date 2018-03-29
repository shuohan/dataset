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

    def create_data(self, *filepaths, types=['none']):
        raise NotImplementedError


class TrainingDataFactory(Data3dFactory):

    def create_data(self, *filepaths, types=['none']):

        data = dict() 

        if 'none' in types:
            none_data = self._create_none(filepaths)
            data['none'] = none_data

            if 'rotation' in types:
                rotated = self._create_rotated(none_data)
                data['rotation'] = rotated

            if 'deformation' in types:
                deformed = self._create_deformed(none_data)
                data['deformation'] = deformed

            if 'flipping' in types:
                flipped = self._create_flipped(none_data)
                data['flipping'] = flipped

                if 'rotation' in types:
                    rotated_flipped = self._create_rotated(flipped)
                    data['flipping_rotation'] = rotated_flipped

                if 'deformation' in types:
                    deformed_flipped = self._create_deformed(flipped)
                    data['flipping_deformation'] = deformed_flipped

        return data

    def _create_none(self, filepaths):
        image = Data3d(filepaths[0], self.get_data_on_the_fly, self.transpose4d)
        label = Data3d(filepaths[1], self.get_data_on_the_fly, self.transpose4d)
        return image, label

    def _create_rotated(self, data):
        rotator = Rotator(max_angle=self.max_angle)
        image = Interpolating3d(data[0], rotator, order=1,
                                get_data_on_the_fly=True)
        label = Interpolating3d(data[1], rotator, order=0,
                                get_data_on_the_fly=True)
        return image, label

    def _create_deformed(self, data):
        shape = data[0].get_data().shape[-3:]
        deformer = Deformer(shape, self.sigma, self.scale)
        image = Interpolating3d(data[0], deformer, order=1,
                                get_data_on_the_fly=True)
        label = Interpolating3d(data[1], deformer, order=0,
                                get_data_on_the_fly=True)
        return image, label

    def _create_flipped(self, data):
        flipper = Flipper(dim=self.dim)
        image = Flipping3d(data[0], flipper, label_pairs=[],
                           get_data_on_the_fly=self.get_data_on_the_fly)
        label = Flipping3d(data[1], flipper, label_pairs=self.label_pairs,
                           get_data_on_the_fly=self.get_data_on_the_fly)
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
        masks = dict()
        if 'none' in types:
            mask = self._create_none(filepaths)
            masks['none'] = mask
            if 'rotation' in types:
                rotated_mask = self._create_rotated(mask, data['rotation'][0])
                masks['rotation'] = rotated_mask

            if 'deformation' in types:
                deformed_mask = self._create_deformed(mask, data['deformation'][0])
                masks['deformation'] = deformed_mask

            if 'flipping' in types:
                flipped_mask = self._create_flipped(mask, data['flipping'][0])
                masks['flipping'] = flipped_mask

                if 'rotation' in types:
                    rotated_flipped_mask = self._create_rotated(flipped_mask,
                                                                data['flipping_rotation'][0])
                    masks['flipping_rotation'] = rotated_flipped_mask

                if 'deformation' in types:
                    deformed_flipped_mask = self._create_deformed(flipped_mask,
                                                                  data['flipping_deformation'][0])
                    masks['flipping_deformation'] = deformed_flipped_mask

        cropped_data = list()
        for key in sorted(data.keys()):
            dd = data[key]
            mask = masks[key]
            cropped = [Cropping3d(d, mask, self.cropping_shape,
                                  d.get_data_on_the_fly) for d in dd]
            cropped_data.append(cropped)

        return cropped_data

    def _create_none(self, filepaths):
        mask = Data3d(filepaths[-1], self.factory.get_data_on_the_fly,
                      self.factory.transpose4d)
        return mask

    def _create_rotated(self, mask, data):
        rotator = data.transformer
        rotated = Interpolating3d(mask, rotator, True)
        return rotated

    def _create_deformed(self, mask, data):
        shape = mask.get_data().shape[-3:]
        deformer = data.transformer
        deformed = Interpolating3d(mask, deformer, True)
        return deformed

    def _create_flipped(self, mask, data):
        flipper = data.transformer
        get_data_on_the_fly = self.factory.get_data_on_the_fly
        flipped = Flipping3d(mask, flipper, label_pairs=[],
                             get_data_on_the_fly=get_data_on_the_fly)
        return flipped
