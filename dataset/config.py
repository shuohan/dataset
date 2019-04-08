# -*- coding: utf-8 -*-

import os
import json
from config import Config_


class Config(Config_):
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
        flip_dim (int): The flipping axis (1 is x axis, etc. channel first)
        image_shape (list of int): The shape of the images to resize to
        crop_shape (list of int): The cropping shape of ROI using mask
        patch_shape (list of int): The shape of extracted patch
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
        super().__init__(config_json)
        self._set_default('max_trans', 30)
        self._set_default('max_rot_angle', 15)
        self._set_default('max_scale', 2)
        self._set_default('def_sigma', 5)
        self._set_default('def_scale', 8)
        self._set_default('sig_int_klim', (10.0, 20.0))
        self._set_default('sig_int_blim', (-1, 1))
        self._set_default('sig_int_num', 5)
        self._set_default('flip_dim', 1)
        self._set_default('image_shape', [196, 256, 196])
        self._set_default('crop_shape', [160, 96, 96])
        self._set_default('patch_shape', [64, 64, 64])
        self._set_default('aug_prob', 1)
        self._set_default('image_suffixes', ['image'])
        self._set_default('label_suffixes', ['label'])
        self._set_default('hierachical_label_suffixes', ['label'])
        self._set_default('mask_suffixes', ['mask'])
        self._set_default('bbox_suffixes', ['bbox', 'mask'])
        self._set_default('label_desc', 'labels.json')
        self._set_default('label_hierachy', 'hierachy.json')
        self._set_default('mask_label_val', 1)
        self._set_default('verbose', False)
        addons = ['resizing', 'flipping', 'cropping', 'label_normalization',
                  'patch', 'mask_extraction']
        augs = ['translation', 'rotation', 'scaling', 'deformation',
                'sigmoid_intensity']
        self._set_default('total_addon', addons)
        self._set_default('total_aug', augs)
        self._set_default('dataset_type', 'dataset')
