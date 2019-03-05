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

    """
    def __init__(self, config_json='configs.json'):
        self._loaded = self._load_json(config_json)
        self._set_default('max_trans', 30)
        self._set_default('max_rot_angle', 15)
        self._set_default('max_scale', 2)
        self._set_default('def_sigma', 5)
        self._set_default('def_scale', 8)
        self._set_default('flip_dim', 1)
        self._set_default('binarize', True)
        self._set_default('image_shape', [256, 256, 256])
        self._set_default('crop_shape', [128, 96, 96])
        self._set_default('aug_prob', 0.5)
        self._set_default('total_addon', ['flipping', 'cropping'])
        self._set_default('total_aug', ['translation', 'rotation', 'scaling',
                                        'deformation'])

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
