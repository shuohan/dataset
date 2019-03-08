# -*- coding: utf-8 -*-

import os
import json
from py_singleton import Singleton


class Config(metaclass=Singleton):
    """Global configurations

    Attributes:
        max_trans (int): The maximum translation for augmentation
        max_rot_angle (float): The maximum rotation angle for augmentation
        max_scale (float): The maximum scaling factor for augmentation
        def_sigma (float): The sigma for generating random deforamtion
        def_scale (float): The scale for generating random deforamtion
        sig_int_klim (tuple of float): The low and high values for k in random
            mixture of sigmoid intensity transformation
        sig_int_blim (tuple of float): The low and high values for b in random
            mixture of sigmoid intensity transformation
        sig_int_num (int): The number of sigmoids of the mixture for intensity
        flip_dim (int): The flipping axis (0 is x axis, etc.)
        crop_shape (list of int): The cropping shape of ROI using mask
        aug_prob (float): The augmentation probability; 1 means always using
            augmentation, 0 means not using
        image_suffixes (list of str): The suffixes of image filenames
        label_suffixes (list of str): The suffixes of label image filenames
        mask_suffixes (list of str): The suffixes of ROI mask image filenames
        bbox_suffixes (list of str): The suffixes of bounding box filenames
        label_desc (str): The basename of the label description .json file in
            the image directory
        total_addon (list of str): All add-on image operations
        total_aug (list of str): All data augmentation operations
        verbose (bool): Print info if True

    """
    def __init__(self, config_json='configs.json'):
        self._loaded = self._load_json(config_json)
        self._set_default('max_trans', 30)
        self._set_default('max_rot_angle', 15)
        self._set_default('max_scale', 2)
        self._set_default('def_sigma', 5)
        self._set_default('def_scale', 8)
        self._set_default('sig_int_klim', (10.0, 20.0))
        self._set_default('sig_int_blim', (-1, 1))
        self._set_default('sig_int_num', 5)
        self._set_default('flip_dim', 0)
        self._set_default('image_shape', [256, 256, 256])
        self._set_default('crop_shape', [160, 96, 96])
        self._set_default('aug_prob', 1)
        self._set_default('image_suffixes', ['image'])
        self._set_default('label_suffixes', ['label'])
        self._set_default('mask_suffixes', ['mask'])
        self._set_default('bbox_suffixes', ['bbox', 'mask'])
        self._set_default('label_desc', 'labels.json')
        self._set_default('total_addon', ['flipping', 'cropping',
                                          'label_normalization'])
        self._set_default('total_aug', ['translation', 'rotation', 'scaling',
                                        'deformation', 'sigmoid_intensity'])
        self._set_default('verbose', False)


    def load(self, config_json):
        """Load .json configurations

        Args:
            config_json (str): The filepath to the configuration .json file

        Raises:
            IndexError: .json file has unsupported configurations

        """
        loaded = self._load_json(config_json)
        for key, value in loaded.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise IndexError('Configuration does not have field %s' % key)

    def _set_default(self, key, default):
        """Set the default value if the setting is not in the loaded json file

        Args:
            key (str): The attribute name
            default (anything): The default value of this attribute

        """
        value = self._loaded[key] if key in self._loaded else default
        setattr(self, key, value)

    def _load_json(self, filename):
        """Load json from file

        Args:
            filename (str): The path to the file to load

        """
        loaded = dict()
        if os.path.isfile(filename):
            with open(filename) as json_file:
                loaded = json.load(json_file)
        return loaded
