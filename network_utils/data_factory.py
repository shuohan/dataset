# -*- coding: utf-8 -*-

from .data import Data3d
from .data_decorators import Cropping3d, Transforming3d
from .transformers import Flipper, Rotator, Deformer


class Data3dFactory:

    def __init__(self, dim=1, label_pairs=[], max_angle=10, sigma = 5, scale=8,
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
                self.image_rotator1 = Rotator(max_angle=self.max_angle, order=1)
                self.label_rotator1 = Rotator(max_angle=self.max_angle, order=0)
                self.image_rotator1.share(self.label_rotator1)
                rotated_image1 = Transforming3d(image, self.image_rotator1,
                                                True)
                rotated_label1 = Transforming3d(label, self.label_rotator1,
                                                True)
                data.append((rotated_image1, rotated_label1))

            if 'deformation' in types:
                shape = image.get_data().shape
                self.image_deformer1 = Deformer(shape, self.sigma, self.scale,
                                                order=1)
                self.label_deformer1 = Deformer(shape, self.sigma, self.scale,
                                                order=0)
                self.image_deformer1.share(self.label_deformer1)
                deformed_image1 = Transforming3d(image, self.image_deformer1,
                                                 True)
                deformed_label1 = Transforming3d(label, self.label_deformer1,
                                                 True)
                data.append((deformed_image1, deformed_label1))

            if 'flipping' in types:
                image_flipper = Flipper(dim=self.dim)
                label_flipper = Flipper(dim=self.dim,
                                        label_pairs=self.label_pairs)
                flipped_image = Transforming3d(image, image_flipper,
                                               self.get_data_on_the_fly)
                flipped_label = Transforming3d(label, label_flipper,
                                               self.get_data_on_the_fly)
                data.append((flipped_image, flipped_label))

                if 'rotation' in types:
                    self.image_rotator2 = Rotator(max_angle=self.max_angle,
                                                  order=1)
                    self.label_rotator2 = Rotator(max_angle=self.max_angle,
                                                  order=0)
                    self.image_rotator2.share(self.label_rotator2)
                    rotated_image2 = Transforming3d(flipped_image,
                                                    self.image_rotator2, True)
                    rotated_label2 = Transforming3d(flipped_label,
                                                    self.label_rotator2, True)
                    data.append((rotated_image2, rotated_label2))

                if 'deformation' in types:
                    shape = image.get_data().shape
                    self.image_deformer2 = Deformer(shape, self.sigma,
                                                    self.scale, order=1)
                    self.label_deformer2 = Deformer(shape, self.sigma,
                                                    self.scale, order=0)
                    self.image_deformer2.share(self.label_deformer2)
                    deformed_image2 = Transforming3d(flipped_image,
                                                     self.image_deformer2, True)
                    deformed_label2 = Transforming3d(flipped_label,
                                                     self.label_deformer2, True)
                    data.append((deformed_image2, deformed_label2))

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
                mask_rotator1 = Rotator(max_angle=self.factory.max_angle,
                                        order=0)
                self.factory.image_rotator1.share(mask_rotator1)
                rotated_mask1 = Transforming3d(mask, mask_rotator1, True)
                masks.append(rotated_mask1)

            if 'deformation' in types:
                shape = mask.get_data().shape
                mask_deformer1 = Deformer(shape, self.factory.sigma,
                                          self.factory.scale, order=0)
                self.factory.image_deformer1.share(mask_rotator1)
                deformed_mask1 = Transforming3d(mask, mask_deformer1, True)
                masks.append(deformed_mask1)

            if 'flipping' in types:
                mask_flipper = Flipper(dim=self.factory.dim)
                flipped_mask = Transforming3d(mask, mask_flipper,
                                              self.factory.get_data_on_the_fly)
                masks.append(flipped_mask)

                if 'rotation' in types:
                    mask_rotator2 = Rotator(max_angle=self.factory.max_angle,
                                            order=0)
                    self.factory.image_rotator2.share(mask_rotator2)
                    rotated_mask2 = Transforming3d(flipped_mask, mask_rotator2,
                                                   True)
                    masks.append(rotated_mask2)

                if 'deformation' in types:
                    shape = mask.get_data().shape
                    mask_deformer2 = Deformer(shape, self.factory.sigma,
                                              self.factory.scale, order=0)
                    self.factory.image_deformer2.share(mask_deformer2)
                    deformed_mask2 = Transforming3d(flipped_mask,
                                                    mask_deformer2, True)
                    masks.append(deformed_mask2)

        cropped_data = list()
        for dd, mask in zip(data, masks):        
            cropped = [Cropping3d(d, mask, self.cropping_shape,
                                  d.get_data_on_the_fly) for d in dd]
            cropped_data.append(cropped)

        return cropped_data
