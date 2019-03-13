# -*- coding: utf-8 -*-
"""Implement Datasets to handel image iteration

"""
import numpy as np
from collections import defaultdict

from .config import Config


class Dataset:
    """Dataset for yielding data

    The length of the dataset will be the number of pipelines times the number
    of images. Hold multiple pipelines and each separately processes all the
    images

    Attributes:
        images (.images.ImageCollection): The collection of images
        verbose (bool): Print info
        pipelines (list of .pipelines.RandomPipeline): Pipeines to process
            images

    """
    def __init__(self, images):
        self.images = images
        self.verbose = Config().verbose
        self.pipelines = list()

    def add_pipeline(self, *pipelines):
        """Add pipelines for image processing

        Args:
            pipeline (.pipelines.RandomPipeline): A pipeline to process images

        """
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
