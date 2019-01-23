# -*- coding: utf-8 -*-

from .data import Transforming3d, Interpolating3d, Flipping3d
from .transformers import Flipper, Translater, Rotater, Deformer, Scaler


def create_aug_strat(augmentation, **kwargs):
    """Create an AugmentationStrategy
    
    Args:
        augmentation (str): The type of the augmentation

    Returns:
        result (AugmentationStrategy): The created augmentation strategy

    """
    if augmentation == 'flipping':
        return FlippingStrategy(**kwargs)
    elif augmentation == 'translation':
        return TranslationStrategy(**kwargs)
    elif augmentation == 'rotation':
        return RotationStrategy(**kwargs)
    elif augmentation == 'scaling':
        return ScalingStrategy(**kwargs)
    elif augmentation == 'deformation':
        return DeformationStrategy(**kwargs)


class AugmentationStrategy:
    """Abstract class for augmentation to apply to the data
    
    """
    def __init__(self, **kwargs):
        pass

    def augment(self, *data):
        """Apply augmentation

        Args:
            data (.data.Data or .data.DataDecorator): The data to augment

        Returns:
            augmentated (list of .data.DataDecorator): The augmented data

        """
        raise NotImplementedError


class RotationStrategy(AugmentationStrategy):
    """Apply rotation augmentation
    
    Args:
        rotater (.transformers.Rotater): Rotate data with the same parameters

    """
    def __init__(self, max_angle=15, point=None):
        self.rotater = Rotater(max_angle=max_angle, point=point)

    def change_rotation_center(self, point):
        """Change the rotation center

        Args:
            points (None or numpy.array): The rotation center

        """
        self.rotater.point = point

    def augment(self, *data):
        data = [Interpolating3d(d, self.rotater, on_the_fly=True) for d in data]
        return data


class TranslationStrategy(AugmentationStrategy):
    """Translate the data

    Args:
        translater (.transformers.Translater): Translate the data with the same
            parameters
    
    """
    def __init__(self, max_trans=30):
        self.translater = Translater(max_trans=max_trans)

    def augment(self, *data):
        dd = [Transforming3d(d, self.translater, on_the_fly=True) for d in data]
        return dd


class FlippingStrategy(AugmentationStrategy):
    def augment(self, *data): 
        pass


class DeformationStrategy(AugmentationStrategy):
    def augment(self, *data):
        pass


class ScalingStrategy(AugmentationStrategy):
    def augment(self, *data):
        pass
