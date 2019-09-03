# -*- coding: utf-8 -*-

import os
import json
from config import Config


class Config(Config):
    """Global configurations.

    Update a configuration via

    >>> Config.attribute = new_value

    Note:
        The configurations are stored as class attributes. Do not use
        initialization to update them.

    """
    max_trans = 30
    """int: The maximum translation for augmentation."""
    max_rot_angle = 15
    """float: The maximum rotation angle for augmentation."""
    max_scale = 2
    """float: The maximum scaling factor for augmentation."""
    def_sigma = 5
    """float: The sigma for generating random deforamtion."""
    def_scale = 8
    """float: The scale for generating random deforamtion."""
    sig_int_klim = (10.0, 20.0)
    """tuple[float]: The low and high values for k in random mixture of sigmoid
    intensity transformation."""
    sig_int_blim = (-1, 1)
    """tuple[float]: The low and high values for b in random mixture of sigmoid
    intensity transformation."""
    sig_int_num = 5
    """int: The number of sigmoids of the mixture for intensity."""
    flip_dim = 1
    """int: The flipping axis (1 is x axis, etc. channel first)."""
    image_shape = (196, 256, 196)
    """tuple[int]: The shape of the images to resize to."""
    crop_shape = (160, 96, 96)
    """tuple[int]: The cropping shape of ROI using mask."""
    patch_shape = (64, 64, 64)
    """tuple[int]: The shape of extracted patch."""
    num_patches = 10
    """int: The number of patches to extract per image."""
    num_slices = 10
    """int: The number of slices to extract per image."""
    slice_dim = -1
    """int: The slice dimension."""
    aug_prob = 0.5
    """float: The augmentation probability; 1 means always using augmentation, 0
    means not using at all."""
    image_suffixes = ['image']
    """list[str]: The suffixes of image filenames."""
    label_suffixes = ['label']
    """list[str]: The suffixes of label image filenames."""
    mask_suffixes = ['mask']
    """list[str]: The suffixes of ROI mask image filenames."""
    bbox_suffixes = ['bbox', 'mask']
    """list[str]: The suffixes of bounding box filenames."""
    label_desc = 'labels.json'
    """str: The basename of the label description .json file in the image
    directory."""
    mask_label_val = 1
    """int: The value used to extract a mask from the label image."""
    drop_ind = -1
    """int: The index of images to drop."""
    verbose = False
    """bool: Print info if True."""
    worker_types = {'addon': ['resize', 'flip', 'crop', 'norm_label', 'drop',
                              'extract_mask', 'extract_patches', 'zero_out',
                              'extract_slices', 'convert_dim', 'zscore'],
                    'aug': ['translate', 'rotate', 'scale', 'deform']}
    """dict: * **addon** (*list[str]*) - Addon image operations.
    * **aug** (*list[str]*) - Augmentation operations."""
