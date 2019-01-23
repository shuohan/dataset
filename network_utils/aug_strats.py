# -*- coding: utf-8 -*-

from .data import Transforming3d, Interpolating3d, Flipping3d
from .transformers import Flipper, Translater, Rotater, Deformer, Scaler
from .configs import Config

config = Config()


def create_aug_strat(augmentation, **kwargs):
    """Create an AugmentationStrategy
    
    Args:
        augmentation (str): The type of the augmentation

    Returns:
        result (AugmentationStrategy): The created augmentation strategy

    """
    if augmentation == 'flipping':
        return FlippingStrategy()
    elif augmentation == 'translation':
        return TranslationStrategy()
    elif augmentation == 'rotation':
        return RotationStrategy()
    elif augmentation == 'scaling':
        return ScalingStrategy()
    elif augmentation == 'deformation':
        return DeformationStrategy()


class AugmentationStrategy:
    """Abstract class for augmentation to apply to the data

    Pass all the data at the same time to AugmentationStrategy.augment to
    perform the transformation with the same parameters

    """
    def augment(self, *data):
        """Apply augmentation

        Args:
            data (.data.Data or .data.DataDecorator): The data to augment

        Returns:
            augmentated (list of .data.DataDecorator): The augmented data

        """
        raise NotImplementedError


class RotationStrategy(AugmentationStrategy):
    """Apply rotation augmentation with the same parameters"""
    def augment(self, *data):
        rotater = Rotater(max_angle=config.max_angle)
        return [Interpolating3d(d, rotater, on_the_fly=True) for d in data]


class TranslationStrategy(AugmentationStrategy):
    """Translate the data with the same parameters"""
    def augment(self, *data):
        translater = Translater(max_trans=config.max_trans)
        return [Transforming3d(d, translater, on_the_fly=True) for d in data]


class FlippingStrategy(AugmentationStrategy):
    """Flip the data"""
    def augment(self, *data): 
        flipper = Flipper(dim=config.flip_dim)
        return [Flipping3d(d, flipper, on_the_fly=d.on_the_fly) for d in data]


class DeformationStrategy(AugmentationStrategy):
    """Deform the data with the same parameters"""
    def augment(self, *data):
        shape = data[0].shape
        deformer = Deformer(shape, config.deform_sigma, config.deform_scale)
        return [Interpolating3d(d, deformer, on_the_fly=True) for d in data]


class ScalingStrategy(AugmentationStrategy):
    """Scale the data with the same parameters"""
    def augment(self, *data):
        scaler = Scaler(max_scale=config.max_scale)
        return [Interpolating3d(d, scaler, on_the_fly=True) for d in data]
