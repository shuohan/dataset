# -*- coding: utf-8 -*-

from augmentation_strategies import AugmentationStrategyFactory
from data import Data


class DataGroup:
    """Group data sharing the same operations

    For example, the image and its corresponding label images need to be
    transformed by the exactly same transforamtion (with the same transformation
    parameters)

    Attributes:
        implementor (DataImp): The implementor handling the augmentation
            application logic
        images (list of data.Data): Contain the image data
        labels (list of data.Data): Contain the label image data
        mask (data.Data): The mask used to crop the images/label images
        strategies (list of AugmentationStrategy): An augmentation strategy
            applies the transformation to augment the data

    """
    def __init__(self):
        self.implementor = None
        self.images = list()
        self.labels = list()
        self.mask = list()
        self.strategies = list()

    def set_implementor(self, imp):
        """Set the data group implementor

        Args:
            imp (DataGroupImp): The implementor to set

        """
        self.implementor = imp

    def add_image(self, data):
        """Add an image into the data group

        DataGroup can accept as many images as needed

        Args:
            data (data.Data): data to add

        """
        self.images.append(data)

    def add_label(self, data):
        """Add an label image into the data group

        DataGroup can accept as many label images as needed

        Args:
            data (data.Data): data to add

        """
        self.labels.append(data)

    def add_mask(self, mask):
        """Add an image mask for cropping into the data group

        DataGroup can only accept one (or none) mask

        Args:
            data (data.Data): data to add

        """
        self.mask = [mask]

    def add_augmentation(self, augmenation):
        """Add an augmentation method to be applied to the data

        Args:
            augmentation (str): The allowed augmentaiton

        """
        factory = AugmentationStrategyFactory()
        strategy = factory.create(augmentation, self)
        self.strategies.append(strategy)

    def augment(self):
        raise NotImplementedError

    def get_datagroup(self):
        raise NotImplementedError


class DataGroupImp:
    """Handle DataGroup augmentation application logic

    Args:
        datagroup (DataGroup): Provide augmentation strategies for selection

    """
    def __init__(self, datagroup):
        self.datagroup = datagroup

    def augment(self):
        """Compose the augmentation to augment the data

        Args:
            images (list of data.Data): The images to augment
            labels (list of data.Data): The label images to augment
            mask (list of data.Data): The cropping mask to augment

        Returns:
            a_images (list of data.Decorator/data.Data): The augmented images
            a_labels (list of data.Decorator/data.Data): The augmented labels
            a_mask (list of data.Decorator/data.Data): The augmented mask
        
        """
        raise NotImplementedError


class RandomDataGroupImp(DataGroupImp):
    """Randomly select an augmentation strategy to change the data

    The self.prob specifies the probability of performing an augmentation and
    the augmentation will be chosen from all available 

    Args:
        prob (float): The probability of performing an augmentation

    """
    def __init__(self, datagroup, prob=0.5):
        super().__init__(datagroup)
        self.prob = prob

    def augment(self, images, labels, mask):
        """See DataGroupImp.augment"""
        if np.random.rand <= self.prob:
            strategy = np.random.choice(self.datagroup.strategies)
            a_images, a_labels, a_mask = strategy.augment(images, labels, mask)
        else:
            a_images, a_labels, a_mask = images, labels, mask
        return a_images, a_labels, a_mask


class SerialDataGroupImp(DataGroupImp):
    """Compose all augmentation strategy to apply to the data
    
    """
    def augment(self, images, labels, mask):
        """See DataGroupImp.augment"""
        a_images, a_labels, a_mask = images, labels, mask
        for stg in self.datagroup.strategies:
            a_images, a_labels, a_mask = stg.augment(a_images, a_labels, a_mask)
        return a_images, a_labels, a_mask
