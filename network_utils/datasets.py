# -*- coding: utf-8 -*-
"""Implement Datasets to handel image iteration

"""
import os
import json
import numpy as np
from glob import glob
from collections import defaultdict

from .images import Image, Label, Mask, BoundingBox


class Dataset:

    def __init__(self, images=None, image_suffixes=['image'], verbose=False):
        self.image_suffixes = image_suffixes
        if isinstance(images, defaultdict) and images.default_factory == list:
            self._images = images
        else:
            self._images = defaultdict(list)
        self._pipelines = list()
        self.verbose = verbose

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

    @property
    def pipelines(self):
        return self._pipelines

    def add_pipeline(self, pipeline):
        self.pipelines.append(pipeline)

    def add_pipelines(self, *pipelines):
        self.pipelines.extend(pipelines)

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
        return len(self.images) * len(self.pipelines)

    def __getitem__(self, key):
        """Get item by key

        Indices are arranged as:

            pipeline 1           pipeline 2          pipeline 3      ...
        _________________    _________________   _________________
        |               |    |               |   |               |
        image1 image2 ...    image1 image2 ...   image1 image2 ...

        Args:
            key (int): The index of the item to get

        """
        if len(self) == 0:
            raise IndexError('No images or no pipeline')
        if key >= len(self):
            raise IndexError('Index %d is out of range %d' % (key, len(self)))
        elif key < 0:
            raise IndexError('Index %d is smaller than 0' % (key,))

        pipeline_ind = key // len(self.images)
        image_ind = key % len(self.images)
        pipeline = self.pipelines[pipeline_ind]
        images = list(self.images.values())[image_ind]
        processed = pipeline.process(*images)
        if self.verbose:
            print('-' * 80)
            for p in processed:
                print(p)
            print('-' * 80)
        return [p.output for p in processed]

    def split(self, indicies):
        indicies2 = sorted(list(set(range(len(self.images))) - set(indicies)))
        keys = np.array(list(self.images.keys()))
        keys1 = keys[indicies]
        keys2 = keys[indicies2]
        print(keys1)
        print(keys2)
        images1 = defaultdict(list, {k: self.images[k] for k in keys1})
        images2 = defaultdict(list, {k: self.images[k] for k in keys2})
        dataset1 = Dataset(images=images1, image_suffixes=self.image_suffixes,
                           verbose=self.verbose)
        dataset2 = Dataset(images=images2, image_suffixes=self.image_suffixes,
                           verbose=self.verbose)
        dataset1.add_pipelines(*self.pipelines)
        dataset2.add_pipelines(*self.pipelines)
        return dataset1, dataset2


class DatasetDecorator(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    @property
    def images(self):
        return self.dataset.images

    @property
    def pipelines(self):
        return self.dataset.pipelines

    @property
    def image_suffixes(self):
        return self.dataset.image_suffixes

    @property
    def verbose(self):
        return self.dataset.verbose

    def add_pipeline(self, pipeline):
        self.dataset.pipelines.append(pipeline)

    def add_pipelines(self, *pipelines):
        self.dataset.pipelines.extend(pipelines)

    def add_images(self):
        raise NotImplementedError

    def __str__(self):
        return self.dataset.__str__()

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, key):
        return self.dataset.__getitem__(key)

    def split(self, indicies):
        return self.dataset.split(indicies)


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


class Located(DatasetDecorator):

    def __init__(self, dataset, bbox_suffixes=['bbox', 'mask']):
        super().__init__(dataset)
        self.bbox_suffixes = bbox_suffixes

    def add_images(self, dirname, ext='.nii.gz', id=''):
        self.dataset.add_images(dirname, ext, id)
        for filepath in sorted(glob(os.path.join(dirname, '*' + ext))):
            parts = os.path.basename(filepath).replace(ext, '').split('_')
            name = os.path.join(id, parts[0])
            if parts[-1] in self.bbox_suffixes:
                bbox = BoundingBox(filepath=filepath)
                self.images[name].append(bbox)
