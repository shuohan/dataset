# -*- coding: utf-8 -*-

from collections import OrderedDict

from .aug_strats import create_image_aug_strat, create_label_aug_strat
from .aug_sel import create_selector
from .transformers import create_transformer


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
        self.selector = create_selector()

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
