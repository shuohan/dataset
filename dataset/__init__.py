# -*- coding: utf-8 -*-

from enum import Enum, auto

from .config import Config
from .datasets import Dataset, WrapperDataset
from .images import ImageLoader, ImageCollection
from .pipelines import RandomPipeline


class DatasetType(Enum):
    dataset = auto()
    wrapper_dataset = auto()


class DatasetFactory:
    """Create Dataset instance

    Attributes:
        loader (.images.ImageLoader): Load images

    """
    def __init__(self):
        self._t_images = list()
        self._v_images = list()
        self.image_types = list()
        self.training_operations = list()
        self.validation_operations = list()

    def add_image_type(self, *image_types):
        """Add an image type such as image, label, mask, bounding_box

        The order of the image_types are preserved. This should be called before
        `add_dataset`

        Args:
            image_type (str): The type of the image to add

        """
        self.image_types.extend(image_types)

    def add_dataset(self, dataset_id='data', dirname=None, val_ind=None,
                    t_dirname=None, v_dirname=None):
        """Add a dataset

        If `t_dirname` and `v_dirname` are None, the dataset are loaded via
        `dirname` and split according to `val_ind` into training and validation
        datasets; otherwise, the training and validation datasets are loaded via
        `t_dirname` and `v_dirname`, respectively.

        The image suffixes of the files within the data directory are configured
        via .config.Config. See Config().image_suffixes etc.

        Args:
            id (str): The id of the dataset
            dirname (str): The directory name of the dataset. The image suffixes
                are configured in .config.Config.
            val_ind (list of int): The indicies of validation data within
                the dataset directory `dirname`
            t_dirname (str): The directory of the training dataset
            v_dirname (str): The directory of the validation dataset

        """
        if dirname is not None:
            loader = ImageLoader(dirname, id=dataset_id)
            loader.load(*self.image_types)
            if val_ind is None:
                t_images, v_images = loader.images, ImageCollection()
            else:
                v_images, t_images = loader.images.split(val_ind)
            self._t_images.append(t_images)
            self._v_images.append(v_images)
        elif t_dirname is not None:
            t_loader = ImageLoader(t_dirname, id=dataset_id)
            t_loader.load(*self.image_types)
            self._t_images.append(t_loader.images)
            if v_dirname is not None:
                v_loader = ImageLoader(v_dirname, id=dataset_id)
                v_loader.load(*self.image_types)
                self._v_images.append(v_loader.images)
            else:
                self._v_images.append(ImageCollection())
        else:
            message = 't_dirname/v_dirname and dirname cannot be all None'
            raise RuntimeError(message)

    def add_training_operation(self, *operations):
        """Add an operation to training dataset pipeline
        
        The order of operations are preserved

        operation (str): A add-on operation (such as cropping)
            and augmentation

        """
        self.training_operations.extend(operations)

    def add_validation_operation(self, *operations):
        """Add an operation to training dataset pipeline
        
        The order of operations are preserved

        operation (str): A add-on operation (such as cropping)
            and augmentation

        """
        self.validation_operations.extend(operations)

    def create(self):
        """Create training and validation datsets

        Returns:
            t_dataset (.datasets.Dataset_): The training dataset
            v_dataset (.datasets.Dataset_): The validation dataset

        """
        t_dataset = self._create(self._t_images, self.training_operations)
        v_dataset = self._create(self._v_images, self.validation_operations)
        return t_dataset, v_dataset

    def _create(self, images, operations):
        images = sum(images[1:], images[0])
        if DatasetType[Config().dataset_type] is DatasetType.dataset:
            dataset = Dataset(images)
        elif DatasetType[Config().dataset_type] is DatasetType.wrapper_dataset:
            dataset = WrapperDataset(images)
        pipeline = RandomPipeline()
        pipeline.register(*operations)
        dataset.add_pipeline(pipeline)
        return dataset
