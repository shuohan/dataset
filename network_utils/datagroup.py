# -*- coding: utf-8 -*-

import numpy as np
from collections import OrderedDict

from .transformers import create_transformer
from .configs import Config

config = Config()


class DataItem:
    pass


class Image(DataItem):
    pass


class ItemDecorator(DataItem):
    pass


class LabelImage(ItemDecorator):
    pass


class MasedImage(ItemDecorator):
    pass


class PatchExtracting(ItemDecorator):
    pass


class DataGroup(DataItem):
    """Group data sharing the same operations

    For example, the image and its corresponding label images need to be
    transformed by the exactly same transforamtion (with the same transformation
    parameters)

    Attributes:
        items (list of DataItem): Contain the images/label images
        transformers (dict of .transformers.Transformer): Apply transform
        selector (AugmentationSelector): Select augmentation from the registered
            augmentation pool

    """
    def __init__(self):
        self.items = list()
        self.augmented_items = list()
        self.transformers = OrderedDict()
        self.selector = create_selector(configs.aug_sel)

    def add_item(self, item):
        """Add a data item into the group, accept as many data as needed

        Args:
            item (Item): The data item to add

        """
        self.items.append(item)
        self.augmented_items.append(item)

    def register_augmentation(self, augmenation):
        """Register an augmentation method into the pool

        Args:
            augmentation (str): The allowed augmentaiton

        """
        self.transformers[augmentation] = create_transformer(augmentation)
        for item in items:
            item.register_augmentation(augmentation, transformer)

    def augment(self):
        """Augment the data"""
        selected = self.selector.select(self.transformers.keys())
        for sel in selected:
            self.transformers[sel].update()
        self.augmented_items = [item.augment(selected) for item in self.items]

    def get_data(self):
        """Return the augmented data

        Returns:
            data (tuple of .data.Data): The augmented data

        """
        data = tuple([item.get_data() for item in self.augmented_items])
        for trans in self.transformers.values():
            trans.cleanup()
        return data


def create_selector(type):
    if type == 'random':
        return RandomSelector()
    elif type == 'serial':
        return SerialSelector()


class AugmentationSelector:
    """Abstract class to select augmentation
    
    """
    def select(self, augmentations):
        """Select the augmentation to apply

        Args:
            augmentations (list of str): The candidate augmentation types to
                select from 

        Returns:
            selected (list of str): The selected augmentations

        """
        raise NotImplementedError


class RandomSelector(AugmentationSelector):
    """Randomly select an augmentation method to change the data

    The self.prob specifies the probability of performing an augmentation and
    this augmentation will be chosen from all available augmentaiton

    Attributes:
        prob (float): The probability of perform an augmentation

    """
    def __init__(self):
        self.prob = configs.aug_prob

    def select(self, augmentations):
        selected = list()
        if np.random.rand() <= self.prob:
            selected = [np.random.choice(augmentations)]
        return selected


class SerialSelector(AugmentationSelector):
    """Select all augmentation strategy to apply to the data"""
    def select(self, augmentations):
        return augmentations
