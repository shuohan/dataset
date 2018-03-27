# -*- coding: utf-8 -*-

import numpy as np
from torch.utils.data import Dataset
from image_processing_3d import crop3d, calc_bbox3d, resize_bbox3d


class DecoratedMedicalImageDataset3d:
    
    def __init__(self, medical_image_dataset):
        self.dataset = medical_image_dataset

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.dataset)
    

class TransformedMedicalImageDataset3d(DecoratedMedicalImageDataset3d):

    def __init__(self, medical_image_dataset, transform):
        super().__init__(medical_image_dataset)
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(self.dataset[index])


class CroppedMedicalImageDataset3d(DecoratedMedicalImageDataset3d):
    def __init__(self, medical_image_dataset, cropping_shape):
        super().__init__(medical_image_dataset)
        self.cropping_shape = cropping_shape 

    def __getitem__(self, index):
        data = self.dataset[index]
        cropped_data = [self._crop(d, data[-1]) for d in data[:-1]]
        return cropped_data

    def _crop(self, image, mask):
        bbox = calc_bbox3d(mask)
        resized_bbox = resize_bbox3d(bbox, self.cropping_shape)
        cropped = crop3d(image, resized_bbox)[0]
        return cropped


class BinarizedMedicalImageDataset3d(DecoratedMedicalImageDataset3d):

    def __init__(self, medical_image_dataset, binarizer):
        super().__init__(medical_image_dataset)
        self.binarizer = binarizer

    def __getitem__(self, index):
        data = self.dataset[index]
        source_data = data[0][None, ...]
        target_data = self._binarize(data[1])
        return source_data, target_data

    def _binarize(self, label_image):
        if not hasattr(self.binarizer, 'classes_'):
            self.binarizer.fit(np.unique(label_image))
        result = self.binarizer.transform(label_image)
        result = np.rollaxis(result, -1) # channels first
        return result
