# -*- coding: utf-8 -*-
"""Implement Datasets to handel image iteration

"""
from collections import defaultdict

from .config import Config
from .images import Label, FileSearcher
from .images import ImageLoader, LabelLoader, MaskLoader, BoundingBoxLoader
from .pipelines import RandomPipeline


class DatasetCreator:

    def __init__(self):
        self.image_types = list()
        self.operations = list()
        self.loaders = dict()
        self.dataset = None
        self._datasets = list()

        self.register_loader('image', ImageLoader)
        self.register_loader('label', LabelLoader)
        self.register_loader('mask', MaskLoader)
        self.register_loader('bbox', BoundingBoxLoader)

    def add_image_type(self, *image_types):
        self.image_types.extend(image_types)

    def add_dataset(self, dirname, dataset_id='data'):
        file_searcher = FileSearcher(dirname).search()
        for image_type in self.image_types:
            Loader = self.loaders[image_type]
            images = Loader(file_searcher).load().images
            self._datasets.append({'dataset_id': dataset_id, 'dirname': dirname,
                                   'images': images})

    def register_loader(self, name, Loader):
        self.loaders[name] = Loader

    def remove_loader(self, name):
        if name in self.loaders:
            self.loaders.pop(name)

    def add_operation(self, *operations):
        """Add an operation to training dataset pipeline
        
        The order of operations are preserved

        operation (str): A add-on operation (such as cropping)
            and augmentation

        """
        self.operations.extend(operations)

    def create(self):
        """Create training and validation datsets

        Returns:
            t_dataset (.datasets.Dataset_): The training dataset
            v_dataset (.datasets.Dataset_): The validation dataset

        """
        images = [v['images'] for v in self._datasets]
        images = sum(images[1:], images[0])
        self.dataset = Dataset(images)
        pipeline = RandomPipeline()
        pipeline.register(*self.operations)
        self.dataset.add_pipeline(pipeline)
        return self

    def __str__(self):
        message = list()
        message.append('Image types:')
        message.append('    ' + ', '.join(self.image_types))
        message.append('Operation types:')
        message.append('    ' + ', '.join(self.operations))
        message.append('Dataset:')
        message.append(self.dataset.__str__())
        message.append('Registered loaders:')
        str_len = max([len(key) for key in self.loaders.keys()])
        message.extend([('    %%%ds: %%s' % str_len) % (k, v)
                        for k, v in self.loaders.items()])
        return '\n'.join(message)

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
        self.verbose = Config.verbose
        self.pipelines = list()

    @property
    def labels(self):
        return self._get_labels(self.images)

    @property
    def normalized_labels(self):
        pass

    def _get_labels(self, images):
        labels = defaultdict()
        for image_group in images:
            for image in image_group:
                if type(image) is Label:
                    labels[image.label_info].append(image)
        return labels

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
        processed = self.get_processed_image_group(key)
        return tuple(im.output for im in processed)

    def get_processed_image_group(self, key):
        self._check_key(key)
        pipeline_ind = key // len(self.images)
        image_ind = key % len(self.images)
        pipeline = self.pipelines[pipeline_ind]
        images = self.images.at(image_ind)
        processed = pipeline.process(*images)
        if self.verbose:
            self._print_image(*processed)
        return tuple(processed)

    def _check_key(self, key):
        if len(self) == 0:
            raise IndexError('No images or no pipeline')
        if key >= len(self):
            raise IndexError('Index %d is out of range %d' % (key, len(self)))
        elif key < 0:
            raise IndexError('Index %d is smaller than 0' % (key,))

    def _print_image(self, *images):
        print('-' * 80)
        for p in images:
            print(p)
        print('-' * 80)
