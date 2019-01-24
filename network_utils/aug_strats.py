# -*- coding: utf-8 -*-

from .data import Transforming3d
from .configs import Config


def create_image_aug_strat(augmentation, item, transformer):
    """Create an AugmentationStrategy for an image
    
    Args:
        augmentation (str): The type of the augmentation
        item (.datagorup.DataItem): The data item with the required properties
        transformer (.transformers.Transformer): The shared transformer to
            transform the data

    Returns:
        result (AugmentationStrategy): The created augmentation strategy

    """
    if augmentation == 'flipping':
        return FlippingStrategy(transformer)
    elif augmentation == 'translation':
        return AugmentationStrategy(transformer)
    elif augmentation == 'rotation':
        return ImageInterpStrategy(transformer)
    elif augmentation == 'scaling':
        return ImageInterpStrategy(transformer)
    elif augmentation == 'deformation':
        return ImageInterpStrategy(transformer)


def create_label_aug_strat(augmentation, item, transformer):
    """Create an AugmentationStrategy for a label image

    Args:
        augmentation (str): The type of the augmentation
        item (.datagorup.DataItem): The data item with the required properties
        transformer (.transformers.Transformer): The shared transformer to
            transform the data

    Returns:
        result (AugmentationStrategy): The created augmentation strategy

    """
    if augmentation == 'flipping':
        return FlippingStrategy(transformer, item.label_pairs)
    elif augmentation == 'translation':
        return AugmentationStrategy(transformer)
    elif augmentation == 'rotation':
        return LabelInterpStrategy(transformer)
    elif augmentation == 'scaling':
        return LabelInterpStrategy(transformer)
    elif augmentation == 'deformation':
        return LabelInterpStrategy(transformer)


class AugmentationStrategy:
    """Apply augmentation to the data

    Attributes:
        transformer (.transformers.Transformer): The shared transformer to
            transform the data

    """
    def __init__(self, transformer):
        self.transformer = transformer

    def augment(self, data):
        """Apply augmentation

        Args:
            data (.data.Data): The data to augment

        Returns:
            augmented (.data.DataDecorator): The augmented data

        """
        return Transforming3d(data, self.transformer, on_the_fly=True)


class ImageInterpStrategy(AugmentationStrategy):
    """Apply interpolation to the image"""
    def augment(self, data):
        return Transforming3d(data, self.transformer, on_the_fly=True,
                              order=Config().image_interp_order)


class LabelInterpStrategy(AugmentationStrategy):
    """Apply interpolation to the label image"""
    def augment(self, data):
        return Transforming3d(data, self.transformer, on_the_fly=True,
                              order=Config().label_interp_order)


class FlippingStrategy(AugmentationStrategy):
    """Flip the data
    
    Args:
        label_pairs (list of tuple of int): The label to swap during flipping
        
    """
    def __init__(self, transformer, label_pairs=list()):
        super().__init__(transformer)
        self.label_pairs = label_pairs

    def augment(self, data): 
        return Transforming3d(data, self.transformer,
                              on_the_fly=data.on_the_fly,
                              label_pairs=self.label_pairs)
