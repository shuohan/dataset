# -*- coding: utf-8 -*-

import json
from py_singleton import Singleton


class Config(metaclass=Singleton):
    """Global configurations

    Attributes:
        max_translation (int): The maximum translation for augmentation
        max_rotation_angle (float): The maximum rotation angle for augmentation
        max_scale (float): The maximum scaling factor for augmentation

    """
    def __init__(self, config_json='configs.json'):
        self._loaded = self._load_json(config_json)
        self._set_default('max_translation', 30)
        self._set_default('max_rotation', 15)
        self._set_default('max_scale', 2)
        self._set_default('deform_sigma', 5)
        self._set_default('deform_scale', 8)
        self._set_default('flip_dim', 1)
        self._set_default('binarize', True)

    def _set_default(self, key, default):
        """Set the default value if the setting is not in the loaded json file

        Args:
            key (str): The attribute name
            default (anything): The default value of this attribute

        """
        value = self._loaded[key] if value in self._loaded else default
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
