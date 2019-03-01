# -*- coding: utf-8 -*-
"""Implement Datasets to handel image iteration

"""
import os
import json
from glob import glob
from collections import defaultdict

from .images import Image, Label, Mask


class Dataset:

    def __init__(self, image_suffixes=['image']):
        self.image_suffixes = image_suffixes
        self._images = defaultdict(list)

    @property
    def images(self):
        return self._images

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
        for name, group in self._images.items():
            info.append(name)
            for image in group:
                info.append('    ' + image.__str__())
            info.append('-' * 80)
        return '\n'.join(info)

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, key):
        raise NotImplementedError


class DatasetDecorator(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset
        self.image_suffixes = self.dataset.image_suffixes

    @property
    def images(self):
        return self.dataset.images

    def add_images(self):
        raise NotImplementedError

    def __str__(self):
        return self.dataset.__str__()

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, key):
        return self.dataset.__getitem__(key)


class Delineated(DatasetDecorator):

    def __init__(self, dataset, label_suffixes=['label'], desc_suffix='labels'):
        super().__init__(dataset)
        self.label_suffixes = label_suffixes
        self.desc_suffix = desc_suffix

    def add_images(self, dirname, ext='.nii.gz', id=''):
        self.dataset.add_images(dirname, ext, id)
        desc_paths = glob(os.path.join(dirname, '*'+self.desc_suffix+'.json'))
        if desc_paths:
            labels, pairs = self._load_label_desc(desc_paths[0])
        else:
            labels, pairs = [], []
        for filepath in sorted(glob(os.path.join(dirname, '*' + ext))):
            parts = os.path.basename(filepath).replace(ext, '').split('_')
            name = os.path.join(id, parts[0])
            if parts[-1] in self.label_suffixes:
                label = Label(filepath=filepath, labels=labels, pairs=pairs)
                self.images[name].append(label)
    
    def _load_label_desc(self, filepath):
        with open(filepath) as jfile:
            contents = json.load(jfile)
        return contents['labels'], contents['pairs']


class Masked(DatasetDecorator):

    def __init__(self, dataset, mask_suffixes=['mask']):
        super().__init__(dataset)
        self.mask_suffixes = mask_suffixes

    def add_images(self, dirname, ext='.nii.gz', id=''):
        self.dataset.add_images(dirname, ext, id)
        for filepath in sorted(glob(os.path.join(dirname, '*' + ext))):
            parts = os.path.basename(filepath).replace(ext, '').split('_')
            name = os.path.join(id, parts[0])
            if parts[-1] in self.mask_suffixes:
                mask = Mask(filepath=filepath)
                self.images[name].append(mask)
