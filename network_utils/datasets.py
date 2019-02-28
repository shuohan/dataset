# -*- coding: utf-8 -*-
"""Implement Datasets to handel image iteration

"""
import os
from glob import glob
from collections import defaultdict

from .images import Image


class Dataset:

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, key):
        raise NotImplementedError


class DatasetDecorator(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset


class ImageDataset(Dataset):

    def __init__(self, image_suffixes=['image']):
        self.image_suffixes = image_suffixes
        self.images = defaultdict(list)

    def add_images(self, dirname, ext='.nii.gz', id=''):
        for filepath in sorted(glob(os.path.join(dirname, '*' + ext))):
            parts = os.path.basename(filepath).replace(ext, '').split('_')
            name = os.path.join(id, parts[0])
            if parts[-1] in self.image_suffixes:
                image = Image(filepath=filepath)
                self.images[name].append(image)

    def __str__(self):
        info = list()
        info.append('-' * 80)
        for name, group in self.images.items():
            info.append(name)
            for image in group:
                info.append('    ' + image.__str__())
            info.append('-' * 80)
        return '\n'.join(info)


# class Delineated(DatasetDecorator):
# 
#     def __init__(self, dataset, label_suffixes=['label']):
#         self.dataset = dataset
#         self.label_suffixes = label_suffixes
# 
#     def add_images(self, dirname, ext='.nii.gz', id=''):
#         pass
# 
# 
# class Masked(DatasetDecorator):
# 
#     def __init__(self, dataset, mask_suffixes=['mask']):
#         self.dataset = dataset
#         self.mask_suffixes = mask_suffixes
# 
#     def add_images(self, dirname, ext='.nii.gz', id=''):
#         pass
