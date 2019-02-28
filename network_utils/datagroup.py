# -*- coding: utf-8 -*-

from collections import OrderedDict

from .aug_strats import create_image_aug_strat, create_label_aug_strat
from .aug_sel import create_selector
from .data import Data
from .transformers import create_transformer


class DataGroup:
    """Group data sharing the same operations

    For example, the image and its corresponding label images need to be
    transformed by the exactly same transforamtion (with the same transformation
    parameters)

    The augmentation logic is handled by self.augmetor. The DataGroup is in
    charge of the maintenance of the pool of augmentation methods.

    Attributes:
        data (list of .data.Data): Contain the images/label images
        augmented_data (list of .data.Data): Contain the augmented data
        transformers (dict of .transformers.Transformer): Apply transform
        augmentor (augmentors.Augmentor): Peform augmentation to data

    """
    def __init__(self):
        self.data = list()
        self.augmented_data = list()
        self.transformers = OrderedDict()
        self.augmentor = create_augmentor(self)

    def add_data(self, data):
        """Add a data item into the group, accept as many data as needed

        Args:
            data (.data.Data): The data to add

        """
        self.data.append(data)
        self.augmented_data.append(data)

    def register_augmentation(self, augmenation):
        """Register an augmentation method into the pool

        Args:
            augmentation (str): The allowed augmentaiton

        """
        self.transformers[augmentation] = create_transformer(augmentation)

    def augment(self):
        """Augment the data"""
        self.augmented_data = self.augmentor.augment()

    def get_data(self):
        """Return the augmented data

        Returns:
            data (tuple of numpy.array): The augmented data

        """
        data = tuple([d.get_data() for d in self.augmented_data])
        for trans in self.transformers.values():
            trans.cleanup()
        return data
