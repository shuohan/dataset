# -*- coding: utf-8 -*-

from .aug_strats import AugmentationStrategyFactory
from .transformers import Cropper


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
        factory = AugmentationStrategyFactory()
        strategy = factory.create(augmentation, self)
        self._strategies.append(strategy)

    def _augment(self):
        """Augment the data"""
        return self.implementor.augment(self.data, self.strategies)

    def get_data(self):
        return self._augment()


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
        self.cropping_shape = (128, 96, 96)

    def set_mask(self, mask):
        """Set an image mask for cropping

        Args:
            mask (.data.Data): The mask used to crop the data

        """
        self.mask = mask

    def set_cropping_shape(self, shape):
        """Set the resulted shape after cropping

        Args:
            shape (tuple of int): The tuple of the shape (x, y, z)

        """
        self.cropping_shape = shape

    def _crop(self, data, mask):
        """Crop the data using the mask with the cropping shape

        Args:
            data (list of .data.Data or .data.DataDecorator): The data to crop

        Returns:
            results (list of .data.DataDecorator): The cropped data

        """
        c = Cropper(mask, self.cropping_shape)
        cropped = [Transforming3d(d, c, on_the_fly=d.on_the_fly) for d in data]
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
