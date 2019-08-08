# -*- coding: utf-8 -*-
"""Datasets handeling image iteration.

"""
from collections import defaultdict

from .config import Config
from .images import Label, FileSearcher
from .images import ImageLoader, LabelLoader, MaskLoader, BoundingBoxLoader
from .pipelines import RandomPipeline


class DatasetCreator:
    """Creates an instance of :class:`Dataset`.

    To create a dataset:

    1. Call the method :meth:`register_loader` to add a class of
       :class:`dataset.images.Loader` into the pool for selection. There are
       several loadesr already registered. Use

       >>> print(Dataset())

       to check the registered loaders. This is primarily for adding new types
       of loaders, so this class can find the correct implementation.

    2. Call the method :meth:`add_image_type` to add new types of images to
       load. The types should be a subset of registered loaders.

    3. Call the method :meth:`add_dataset` to concatenate an additional
       directory of data to load.
       
    4. Call the method :meth:`add_operation` to add a new type of operation, for
       augmentation or add-on, to apply to the data.

    5. Call the method :meth:`create` to create the dataset and access it via
       the attribute :attr:`dataset`.
    
    Attributes:
        image_types (list[str]): The types of images to load. Call the method
            :meth:`add_image_type` to add a new type of images to load.
        operations (list[str]): The operations to apply to the data. Call the
            method :meth:`add_operation` to add a new operation.
        loaders (list[dataset.images.Loader]): The registered loaders. Call the
            method :meth:`register_loader` to register.
        dataset (Dataset): The created dataset. ``None`` if the method
            :meth:`create` has not been called.

    """
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
        """Adds a type of images to load.

        Args:
            image_types (str): The type of image to load. It should be in the
                registered loaders.

        """
        self.image_types.extend(image_types)

    def add_dataset(self, dirname, dataset_id='data'):
        """Adds a directory to load.

        Args:
            dirname (str): The path to the directory.
            dataset_id (str): Used to identify datasets loaded from different
                direcories.

        """
        file_searcher = FileSearcher(dirname).search()
        for image_type in self.image_types:
            Loader = self.loaders[image_type]
            images = Loader(file_searcher).load().images
            self._datasets.append({'dataset_id': dataset_id, 'dirname': dirname,
                                   'images': images})

    def register_loader(self, name, Loader):
        """Registers a loader.

        Args:
            name (str): The name for the loader
            Loader (dataset.images.Loader, class): The loader to register.
        
        """
        self.loaders[name] = Loader

    def remove_loader(self, name):
        """Removes a loader from the pool.

        Args:
            name (str): The loader to remove.

        """
        if name in self.loaders:
            self.loaders.pop(name)

    def add_operation(self, *operations):
        """Adds an operation to the processing pipeline.
        
        Note:
            The order of added operations are preserved. The operations should
            be registered in the class :class:`dataset.workers.WorkerCreator`.

        Args:
            operations (str): The name of the operations to add.

        """
        self.operations.extend(operations)

    def create(self):
        """Creates a dataset.

        Note:
            This method does not return an instance of :class:`Dataset`.
            Instead, it returns the instance of :class:`DatasetCreator` itself.
            Use the attribute :attr:`dataset` to access the created dataset.

        Returns:
            DatasetCreator: The instance itself.

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
    """Dataset for yielding data.

    The length of the dataset will be the number of pipelines times the number
    of images. It can hold multiple pipelines to separately processes all the
    images.

    >>> print(dataset) # show all the loaded images.

    The indices of the images are arranged as:

    .. code-block::


            pipeline 1           pipeline 2          pipeline 3      ...
        _________________    _________________   _________________
        |               |    |               |   |               |
        image1 image2 ...    image1 image2 ...   image1 image2 ...


    Attributes:
        images (dataset.images.ImageCollection): The collection of images.
        verbose (bool): Print info if ``True``.
        pipelines (list[dataset.pipelines.RandomPipeline]): Pipelines to process
            the images.

    """
    def __init__(self, images):
        self.images = images
        self.verbose = Config.verbose
        self.pipelines = list()

    @property
    def labels(self):
        """Returns the labels of all data.

        Returns:
            collections.defaultdict: Each key is a unique
            :class:`dataset.images.LabelInfo` and the value is all the images
            that has this set of labels.
        
        """
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
        """Adds pipelines to process the data.

        Args:
            pipeline (dataset.pipelines.RandomPipeline): A processing pipeline.

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
        processed = self.get_processed_image_group(key)
        return tuple(im.output for im in processed)

    def get_processed_image_group(self, key):
        """Returns the processed images at ``key``.

        Returns:
            tuple[dataset.images.Image]: The procesesd images.

        """
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
