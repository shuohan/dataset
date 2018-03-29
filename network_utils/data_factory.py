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

        data = list()

        if 'none' in types:
            image = Data3d(filepaths[0], self.get_data_on_the_fly,
                           self.transpose4d)
            label = Data3d(filepaths[1], self.get_data_on_the_fly,
                           self.transpose4d)
            data.append((image, label))

            if 'rotation' in types:
                self.rotator = Rotator(max_angle=self.max_angle)
                rotated_image = Interpolating3d(image, self.rotator, order=1,
                                                get_data_on_the_fly=True)
                rotated_label = Interpolating3d(label, self.rotator, order=0,
                                                get_data_on_the_fly=True)
                data.append((rotated_image, rotated_label))

            if 'deformation' in types:
                shape = image.get_data().shape[-3:]
                self.deformer = Deformer(shape, self.sigma, self.scale)
                deformed_image = Interpolating3d(image, self.deformer, order=1,
                                                 get_data_on_the_fly=True)
                deformed_label = Interpolating3d(label, self.deformer, order=0,
                                                 get_data_on_the_fly=True)
                data.append((deformed_image, deformed_label))

            if 'flipping' in types:
                self.flipper = Flipper(dim=self.dim)
                flipped_image = Flipping3d(image, self.flipper, [],
                                           self.get_data_on_the_fly)
                flipped_label = Flipping3d(label, self.flipper, self.label_pairs,
                                           self.get_data_on_the_fly)
                data.append((flipped_image, flipped_label))

                if 'rotation' in types:
                    self.flipped_rotator = Rotator(max_angle=self.max_angle)
                    flipped_rotated_image = Interpolating3d(flipped_image,
                                                            self.flipped_rotator,
                                                            1, True)
                    flipped_rotated_label = Interpolating3d(flipped_label,
                                                            self.flipped_rotator,
                                                            0, True)
                    data.append((flipped_rotated_image, flipped_rotated_label))

                if 'deformation' in types:
                    shape = image.get_data().shape[-3:]
                    self.flipped_deformer = Deformer(shape, self.sigma, self.scale)
                    flipped_deformed_image = Interpolating3d(flipped_image,
                                                             self.flipped_deformer,
                                                             1, True)
                    flipped_deformed_label = Interpolating3d(flipped_label,
                                                             self.flipped_deformer,
                                                             0, True)
                    data.append((flipped_deformed_image,
                                 flipped_deformed_label))

        return data


class DecoratedData3dFactory(Data3dFactory):

    def __init__(self, data3d_factory):
        self.factory = data3d_factory


class CroppedData3dFactory(DecoratedData3dFactory):

    def __init__(self, data3d_factory, cropping_shape):
        super().__init__(data3d_factory)
        self.cropping_shape = cropping_shape

    def create_data(self, *filepaths, types=['none']):
        data = self.factory.create_data(*filepaths[:-1], types=types)
        masks = list() 
        if 'none' in types:
            mask = Data3d(filepaths[-1], self.factory.get_data_on_the_fly,
                          self.factory.transpose4d)
            masks.append(mask)

            if 'rotation' in types:
                rotated_mask = Interpolating3d(mask, self.factory.rotator, True)
                masks.append(rotated_mask)

            if 'deformation' in types:
                shape = mask.get_data().shape[-3:]
                deformed_mask = Interpolating3d(mask, self.factory.deformer, True)
                masks.append(deformed_mask)

            if 'flipping' in types:
                flipped_mask = Flipping3d(mask, self.factory.flipper, [],
                                          self.factory.get_data_on_the_fly)
                masks.append(flipped_mask)

                if 'rotation' in types:
                    flipped_rotated_mask = Interpolating3d(flipped_mask,
                                                           self.factory.flipped_rotator,
                                                           0, True)
                    masks.append(flipped_rotated_mask)

                if 'deformation' in types:
                    shape = mask.get_data().shape[-3:]
                    flipped_deformed_mask = Interpolating3d(flipped_mask,
                                                            self.factory.flipped_deformer,
                                                            1, True)
                    masks.append(flipped_deformed_mask)

        cropped_data = list()
        for dd, mask in zip(data, masks):        
            cropped = [Cropping3d(d, mask, self.cropping_shape,
                                  d.get_data_on_the_fly) for d in dd]
            cropped_data.append(cropped)

        return cropped_data
