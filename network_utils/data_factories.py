# -*- coding: utf-8 -*-

from .data import Data3d
from .data_decorators import Cropping3d, Interpolating3d, Flipping3d
from .data_decorators import Binarizing3d
from .transformers import Flipper, Rotator, Deformer


class Data3dFactory:
    """Create Data3d instance

    Call `create` to create Data3d/decorated Data3d instance. It supports
    untouched, flipped, rotated, and deformed processing.

    Attributes:
        dim (int): The flipping dimension. Assume channel first, i.e. dim=0 is
            the channel axis
        label_pairs (list of list of int): The pairs of labels to swap after
            flipping. These are usually corresponding left and right labels. So
            the left labels are still left after flipping.
            Example: [[11, 21], [12, 22], [13, 23]] will swap 11 and 21, 12 and
            22, and 13 and 23.
        max_angel (float): The rotation angles are randomly drawn from uniform
            distribution [-max_angle, max_angle]
        sigma (float): Control smoothness of the random generated deformation
            field. The larget the value is, the smoother the deformation
        scale (float): Draw a value randomly from uniform distribution
            [0, `scale`] to limit the maximum magnitude of the deformation
            pixel-wise displacement.
        get_data_on_the_fly (bool): Get original data and flipped data on the
            fly. The rotated and deformed data are always generated on the fly.
        transpose4d (bool): (Future): Move the last dimension to the first
        types (list of str): {'none', 'flipping', 'rotation', 'deformation'}.
            'none': get original data; 'flipping': flip the data; 'rotation':
            randomly rotate the data; 'deformation': randomly deform the data.
            Note that when flipping and rotation/deformation present at the same
            time, it rotates/deforms the flipped data as well.

        data (dict of tuple .data.Data): The loaded/processed data by current
            call of `self.create`. Each call of `self.create` will
            empty `self.data` first. Key is the processing performed {'none',
            'flipped', 'rotated', 'deformed', 'rotated_flipped',
            'deformed_flipped'}
        
    """
    def __init__(self, dim=1, label_pairs=[], max_angle=10, sigma=5, scale=8,
                 get_data_on_the_fly=False, transpose4d=True, types=['none']):
        self.dim = dim
        self.label_pairs = label_pairs
        self.max_angle = max_angle
        self.sigma = sigma
        self.scale = scale
        self.get_data_on_the_fly = get_data_on_the_fly
        self.transpose4d = transpose4d
        self.types = types
        self.data = dict()

    def create(self, *filepaths):
        """Create data

        Call this method and access `self.data` to get the created data

        Args:
            filepath (str): The filepath of the data to load

        """
        self.data = dict() 
        if 'none' in self.types:
            self._create_none(filepaths)
            if 'rotation' in self.types:
                self._create_rotated()
            if 'deformation' in self.types:
                self._create_deformed()
            if 'flipping' in self.types:
                self._create_flipped()
                if 'rotation' in self.types:
                    self._create_rotated_flipped()
                if 'deformation' in self.types:
                    self._create_deformed_flipped()

    def _create_none(self, filepaths):
        """Abstract method to create untouched data"""
        raise NotImplementedError

    def _create_flipped(self):
        """Abstract method to create flipped data"""
        raise NotImplementedError

    def _create_rotated(self):
        """Abstract method to create rotated data"""
        raise NotImplementedError

    def _create_rotated_flipped(self):
        """Abstract method to create rotated flipped data"""
        raise NotImplementedError

    def _create_deformed(self):
        """Abstract method to create deformed data"""
        raise NotImplementedError

    def _create_deformed_flipped(self):
        """Abstract method to create deformed flipped data"""
        raise NotImplementedError


class TrainingDataFactory(Data3dFactory):
    """Create tranining data

    Call `self.create` and `self.data` contains a list of pairs of (image,
    label).

    Attributes:
        data (dict of tuple of .data.Data): Each item is a pair of (image,
            label)

    """
    def _create_none(self, filepaths):
        image = Data3d(filepaths[0], self.get_data_on_the_fly, self.transpose4d)
        label = Data3d(filepaths[1], self.get_data_on_the_fly, self.transpose4d)
        self.data['none'] = (image, label)

    def _create_flipped(self):
        flipper = Flipper(dim=self.dim)
        image = Flipping3d(self.data['none'][0], flipper, label_pairs=[],
                           get_data_on_the_fly=self.get_data_on_the_fly)
        label = Flipping3d(self.data['none'][1], flipper,
                           label_pairs=self.label_pairs,
                           get_data_on_the_fly=self.get_data_on_the_fly)
        self.data['flipped'] = (image, label)
    
    def _create_rotated(self):
        self.data['rotated'] = self._rotate(self.data['none'])

    def _create_rotated_flipped(self):
        self.data['rotated_flipped'] = self._rotate(self.data['flipped'])

    def _rotate(self, data):
        rotator = Rotator(max_angle=self.max_angle)
        image = Interpolating3d(data[0], rotator, order=1,
                                get_data_on_the_fly=True)
        label = Interpolating3d(data[1], rotator, order=0,
                                get_data_on_the_fly=True)
        return image, label

    def _create_deformed(self):
        self.data['deformed'] = self._deform(self.data['none'])

    def _create_deformed_flipped(self):
        self.data['deformed_flipped'] = self._deform(self.data['flipped'])

    def _deform(self, data):
        shape = data[0].get_data().shape[-3:]
        deformer = Deformer(shape, self.sigma, self.scale)
        image = Interpolating3d(data[0], deformer, order=1,
                                get_data_on_the_fly=True)
        label = Interpolating3d(data[1], deformer, order=0,
                                get_data_on_the_fly=True)
        return image, label


class Data3dFactoryDecorator(Data3dFactory):
    """Decorate Data3dFactory
    
    Attributes:
        factory (Data3dFactory): The factory to decorate
        types (list of int): Same with `self.factory.types`

    """
    def __init__(self, data3d_factory):
        self.factory = data3d_factory
        self.types = self.factory.types

    def create(self, *filepaths):
        self.factory.create(*filepaths)
        super().create(*filepaths)


class Data3dFactoryCropper(Data3dFactoryDecorator):
    """Crop Data3dFactory using corresponding mask

    The mask should be processed in the same way with the data to crop. The
    current implementation only applies the last processing of the data, which
    is correct for now, since only one or two kinds of processing is done to the
    data and the code explicitly selects whether the last processing should be
    done to the original data or flipped data.

    Update: It should work with more transformations. But it is not tested.

    Attributes:
        cropping_shape ((3,) tuple of int): The result shape of the cropped data
        uncropped_data (dict of tuple of .data.Data): Store the uncropped data
            (image, label, mask). Calling `self.create` will emtpy this
            first.

    """
    def __init__(self, data3d_factory, cropping_shape):
        super().__init__(data3d_factory)
        self.uncropped_data = dict()
        self.cropping_shape = cropping_shape

    def create(self, *filepaths):
        self.uncropped_data = dict()
        super().create(*filepaths)

    def _create_none(self, filepaths):
        mask = Data3d(filepaths[-1], self.factory.get_data_on_the_fly,
                      self.factory.transpose4d)
        self._crop('none', mask)

    def _create_flipped(self):
        mask = self._transform('none', 'flipped', [Flipping3d])
        self._crop('flipped', mask)

    def _create_rotated(self):
        mask = self._transform('none', 'rotated', [Interpolating3d])
        self._crop('rotated', mask)

    def _create_rotated_flipped(self):
        mask = self._transform('flipped', 'rotated_flipped', [Interpolating3d])
        self._crop('rotated_flipped', mask)

    def _create_deformed(self):
        mask = self._transform('none', 'deformed', [Interpolating3d])
        self._crop('deformed', mask)

    def _create_deformed_flipped(self):
        mask = self._transform('flipped', 'deformed_flipped', [Interpolating3d])
        self._crop('deformed_flipped', mask)

    def _transform(self, source_key, target_key, Transformings):
        """Transform the corresponding mask
        
        Args:
            source_key: The key of `self.uncropped_data` to perform the
                transformation on
            target_key: The key of transformed result in `self.data` and
                `self.uncropped_data`
            Transformings (list of .data.DataDecorator): The transformation to
                apply. The applying order is the same with the order of items in
                Transformings.

        Returns:
            mask (.data.Data): The transformed mask

        """
        data = self.factory.data[target_key][0]
        datas = list()
        for i in range(len(Transformings)):
            datas.insert(0, data)
            data = data.data
        mask = self.uncropped_data[source_key][-1]
        for Transforming, data in zip(Transformings, datas):
            mask = Transforming(mask, data.transformer,
                                get_data_on_the_fly=data.get_data_on_the_fly)
        return mask

    def _crop(self, key, mask):
        """Crop the mask

        Args:
            key (str): The key of the data to crop in `self.factory.data`
            mask (.data.Data): The mask used to crop the data

        """
        self.uncropped_data[key] = (*self.factory.data[key], mask)
        self.data[key] = tuple([Cropping3d(d, mask, self.cropping_shape,
                                           d.get_data_on_the_fly)
                                for d in self.factory.data[key]])


class Data3dFactoryBinarizer(Data3dFactoryDecorator):
    """Binarize label image

    Args:
        binarizer (.label_image_binarizer.LabelImageBinarizer): Check
            LabelImageBinarizer for more details
        binarizing_indices (list of int): The indices of data tuple to binarize

    """
    def __init__(self, data3d_factory, binarizer, binarizing_indices=[1]):
        super().__init__(data3d_factory)
        self.binarizer = binarizer
        self.binarizing_indices = binarizing_indices

    def _create_none(self, filepaths):
        result = self._binarize(self.factory.data['none'])
        self.data['none'] = result

    def _create_flipped(self):
        result = self._binarize(self.factory.data['flipped'])
        self.data['flipped'] = result

    def _create_rotated(self):
        result = self._binarize(self.factory.data['rotated'])
        self.data['rotated'] = result

    def _create_deformed(self):
        result = self._binarize(self.factory.data['deformed'])
        self.data['deformed'] = result

    def _create_rotated_flipped(self):
        result = self._binarize(self.factory.data['rotated_flipped'])
        self.data['rotated_flipped'] = result

    def _create_deformed_flipped(self):
        result = self._binarize(self.factory.data['deformed_flipped'])
        self.data['deformed_flipped'] = result

    def _binarize(self, data):
        """Binarize data

        Args:
            data (tuple of .data.Data): The data to binarize

        Returns:
            data (tuple of .data.Data): Binarized data

        """
        data = list(data)
        for idx in self.binarizing_indices:
            data[idx] = Binarizing3d(data[idx], self.binarizer,
                                     data[idx].get_data_on_the_fly)
        return tuple(data)
