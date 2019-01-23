# -*- coding: utf-8 -*-

import numpy as np

from .aug_strats import create_aug_strat
from .configs import Config
from .data import Cropping3d
from .transformers import Cropper, LabelImageBinarizer

config = Config()


class DataGroup:
    """Group data sharing the same operations

    For example, the image and its corresponding label images need to be
    transformed by the exactly same transforamtion (with the same transformation
    parameters)

    Attributes:
        implementor (DataGroupImp): The implementor handling the augmentation
            application logic
        data (list of .data.Data): Contain the images/label images
        strategies (list of .aug_strats.AugmentationStrategy): An augmentation
            strategy applies the transformation to augment the data

    """
    def __init__(self):
        self._implementor = None
        self._data = list()
        self._strategies = list()

    @property
    def implementor(self):
        return self._implementor

    @implementor.setter
    def implementor(self, imp):
        self._implementor = imp

    @property
    def data(self):
        return self._data

    def add_data(self, data):
        """Add a data into the data group, accept as many data as needed

        Args:
            data (.data.Data): The data to add

        """
        self._data.append(data)

    @property
    def strategies(self):
        return self._strategies

    def add_augmentation(self, augmenation):
        """Add an augmentation method to be applied to the data

        Args:
            augmentation (str): The allowed augmentaiton

        """
        self._strategies.append(create_aug_strat(augmentation))

    def _augment(self):
        """Augment the data"""
        return self.implementor.augment(self.data, self.strategies)

    def get_data(self):
        return self._augment()


class LabelImageGroup(DataGroup):
    """Data group containing label images

    """
    def __init__(self):
        super().__init__()
        self._label_ind = list()

    def add_data(self, data):
        raise RuntimeError('Use LabelImageGroup.add_image instead')

    def add_image(self, image):
        """Add an image into the datagroup

        Args:
            image (.data.Image3d): The image to add

        """
        self._data.append(data)
        self._label_ind.append(False)

    def add_label(self, image):
        """Add a label image into the datagroup

        Args:
            label (.data.Label3d): The label image to add

        """
        self._data.append(data)
        self._label_ind.append(True)

    def _augment(self):
        """Augment the data"""
        aug = np.array(self.implementor.augment(self.data, self.strategies))
        label_ind = np.array(self._label_ind)
        images = aug[np.logical_not(label_ind)]
        labels = self._binarize(aug[label_ind])
        return (*images, *labels)

    def _binarize(self, labels):
        """One-hot encode the label images"""
        b = LabelImageBinarizer()
        return [Transforming3d(l, b, on_the_fly=l.on_the_fly) for l in labels]


class DataGroupDecorator(DataGroup):
    """Decorate DataGroup

    """
    def __init__(self, datagroup):
        super().__init__()
        self.datagroup = datagroup

    @property
    def implementor(self):
        return self.datagroup.implementor

    @implementor.setter
    def implementor(self, imp):
        raise RuntimeError('Cannot set implementor in DataGroupDecorator')

    @property
    def data(self):
        return self.datagroup.data

    @property
    def strategies(self):
        return self.datagroup.strategies

    def _augment(self):
        raise NotImplementedError


class DataGroupMasking(DataGroupDecorator):
    """Use a mask to crop the data

    Attributes:
        mask (.data.Data): The mask to crop the data
        cropping_shape (tuple of int): The tuple of shape of the cropped

    """
    def __init__(self, datagroup):
        super().__init__(datagroup)
        self.mask = None

    def set_mask(self, mask):
        """Set an image mask for cropping

        Args:
            mask (.data.Data): The mask used to crop the data

        """
        self.mask = mask

    def _crop(self, data, mask):
        """Crop the data using the mask with the cropping shape

        Args:
            data (list of .data.Data or .data.DataDecorator): The data to crop

        Returns:
            results (list of .data.DataDecorator): The cropped data

        """
        c = Cropper(mask, config.cropping_shape)
        cropped = [Cropping3d(d, c, on_the_fly=d.on_the_fly) for d in data]
        return cropped

    def _augment(self):
        data = (*self.data, self.mask)
        augmented = self.implementor.augment(data, self.strategies)
        return self._crop(augmented[:-1], augmented[-1])


class DataGroupExtracting(DataGroupDecorator):
    """Extract patches from DataGroup

    """
    def __init__(self, datagroup, num_patches=10):
        super().__init__(datagroup)
        self.num_patches = num_patches

    def _augment(self):
        raise NotImplementedError


class DataGroupImp:
    """Handle DataGroup augmentation application logic

    Args:
        datagroup (DataGroup): Provide augmentation strategies for selection

    """
    def augment(self, data, strategies):
        """Compose the augmentation to augment the data

        Args:
            data (list of .data.Data): The data to augment
            strategies (list of aug_strats.AugmentationStrategy): The
                augmentatiion strategies to apply to the data
                
        """
        raise NotImplementedError


class RandomDataGroupImp(DataGroupImp):
    """Randomly select an augmentation strategy to change the data

    The self.prob specifies the probability of performing an augmentation and
    the augmentation will be chosen from all available 

    Args:
        prob (float): The probability of performing an augmentation

    """
    def __init__(self, prob=0.5):
        self.prob = prob

    def augment(self, data, strategies):
        if np.random.rand <= self.prob:
            strategy = np.random.choice(strategies)
            data = strategy.augment(*data)
        return data


class SerialDataGroupImp(DataGroupImp):
    """Compose all augmentation strategy to apply to the data
    
    """
    def augment(self, data, strategies):
        for strategy in strategies:
            data = strategy.augment(*data)
        return data
