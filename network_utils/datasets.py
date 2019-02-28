# -*- coding: utf-8 -*-
"""Implement Datasets to handel image iteration

"""
import os
import json
from glob import glob
from collections import defaultdict

from .images import Image, Label


class Dataset:

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, key):
        raise NotImplementedError

    @property
    def images(self):
        return self._images


class DatasetDecorator(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    @property
    def images(self):
        return self.dataset._images


class ImageDataset(Dataset):

    def __init__(self, image_suffixes=['image']):
        self.image_suffixes = image_suffixes
        self._images = defaultdict(list)

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


class Delineated(DatasetDecorator):

    def __init__(self, dataset, label_suffixes=['label'], desc_suffix='labels'):
        self.dataset = dataset
        self.label_suffixes = label_suffixes
        self.desc_suffix = desc_suffix
        self.image_suffixes = dataset.image_suffixes

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
    
    def __str__(self):
        return self.dataset.__str__()

    def _load_label_desc(self, filepath):
        with open(filepath) as jfile:
            contents = json.load(jfile)
        return contents['labels'], contents['pairs']


class Masked(DatasetDecorator):

    def __init__(self, dataset, mask_suffixes=['mask']):
        self.dataset = dataset
        self.mask_suffixes = mask_suffixes

    def add_images(self, dirname, ext='.nii.gz', id=''):
        pass
