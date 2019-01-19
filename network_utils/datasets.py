# -*- coding: utf-8 -*-

import os
from glob import glob

from .data_factories import Data3dFactoryCropper, Data3dFactoryBinarizer


class Dataset3d:
    """Dataset of 3D or 4D (multi-channel 3D) images

    Attributes:
        
    """
    def __init__(self, data):
        """Default initialize

        Args:
            data (list of list of data.Data): The data to handle

        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        self.data[index][0].update()
        data = [d.get_data() for d in self.data[index]]
        self.data[index][0].cleanup()
        return data

    def __add__(self, dataset):
        """Combine with other dataset

        Args:
            dataset (Dataset3d): The dataset to combine
        
        Returns:
            combined_dataset (Dataset3d): The combined datasets

        """
        combined_data = self.data + dataset.data
        return Dataset3d(combined_data)

    def split(self, indices):
        """Split self to two datasets

        Args:
            indices (list of int): The indices in the original data of the first
                split dataset

        """
        indices_all = set(range(len(self)))
        indices1 = set(indices)
        indices2 = indices_all - indices1
        data1 = [self.data[i] for i in sorted(list(indices1))]
        data2 = [self.data[i] for i in sorted(list(indices2))]
        return Dataset3d(data1), Dataset3d(data2)


class Dataset3dFactory:
    """Create Dataset3d instance

    """
    @classmethod
    def create(cls, data_factory, validation_indices, image_paths, label_paths,
               mask_paths=[], cropping_shape=[], binarizer=None,
               include_none=True, include_flipped=True):
        """Create Dataset3d instance

        Args:
            data_factory (.data_factories.Data3dFactory): Factory to create
                Data3d instances
            validation_indices (list of int): The indices of the data from paths
                to treat as validation
            image_paths (list of str): The paths to the training images
            label_paths (list of str): The paths to the label images
            mask_paths (list of str): The paths to the cropping masks. A
                bounding box surrounding the mask will be used to crop the
                images and label images
            cropping_shape ((3,) tuple of int): The shape of the bounding box
        
        """
        if len(mask_paths) > 0 and len(cropping_shape) > 0:
            data_factory = Data3dFactoryCropper(data_factory, cropping_shape)
        if binarizer is not None:
            data_factory = Data3dFactoryBinarizer(data_factory, binarizer)

        data = list()
        if len(mask_paths) > 0:
            for ip, lp, mp in zip(image_paths, label_paths, mask_paths):
                data_factory.create(ip, lp, mp)
                data.append(data_factory.data)
        else:
            for ip, lp in zip(image_paths, label_paths):
                data_factory.create(ip, lp)
                data.append(data_factory.data)
        
        datasets = {key: Dataset3d([d[key] for d in data])
                    for key in data[0].keys()}

        if len(datasets) == 0:
            return None
        
        val_dataset, train_dataset = datasets['none'].split(validation_indices)
        datasets.pop('none')
        if not include_none:
            train_dataset = Dataset3d([])
        if 'flipped' in datasets and not include_flipped:
            datasets.pop('flipped')
        for key, dataset in datasets.items():
            vd, td = dataset.split(validation_indices)
            train_dataset = train_dataset + td
        return train_dataset, val_dataset
