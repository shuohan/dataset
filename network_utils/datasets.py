# -*- coding: utf-8 -*-
"""Implement Datasets to handel image iteration

"""
import numpy as np
from collections import defaultdict


class Dataset:

    def __init__(self, images, verbose=False):
        self.images = images
        self.verbose = verbose
        self.pipelines = list()

    def add_pipeline(self, *pipelines):
        self.pipelines.extend(pipelines)

    def __str__(self):
        info = list()
        info.append('-' * 80)
        for name, group in self.images.items():
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
        images1 = defaultdict(list, {k: self.images[k] for k in keys1})
        images2 = defaultdict(list, {k: self.images[k] for k in keys2})
        dataset1 = Dataset(images=images1, verbose=self.verbose)
        dataset2 = Dataset(images=images2, verbose=self.verbose)
        dataset1.add_pipeline(*self.pipelines)
        dataset2.add_pipeline(*self.pipelines)
        return dataset1, dataset2
