# -*- coding: utf-8 -*-

import numpy as np

from .configs import Config


def create_selector():
    if Config().aug_sel == 'random':
        return RandomSelector()
    elif Config().aug_sel == 'serial':
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
        self.prob = Config().aug_prob

    def select(self, augmentations):
        selected = list()
        if np.random.rand() <= self.prob:
            selected = [np.random.choice(augmentations)]
        return selected


class SerialSelector(AugmentationSelector):
    """Select all augmentation strategy to apply to the data"""
    def select(self, augmentations):
        return augmentations
