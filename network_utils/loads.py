# -*- coding: utf-8 -*-
"""Functions loading data from hard drive to memory

"""
import os
import json


def load(filepath):
    """Wrapper

    """
    if filepath.endswith('.npy'):
        return _load_npy(filepath)
    elif filepath.endswith('.nii') or filepath.endswith('.nii.gz'):
        return _load_nii(filepath)


def _load_npy(filepath):
    """Load .npy file
    
    Args:
        filepath (str): The path to the file to load

    Returns:
        data (numpy.array): Loaded data

    """
    import numpy as np
    return np.load(filepath)


def _load_nii(filepath):
    """Load .nii/.nii.gz file

    Args:
        filepath (str): The path to the file to load

    Returns:
        data (numpy.array): Loaded data

    """
    import nibabel as nib
    return nib.load(filepath).get_data()


def load_label_desc(filepath):
    """Load label description

    Args:
        filepath (str): The path to the description json file

    Returns:
        labels (dict): The label value and name
        pairs (list of list): Each is a pair of left/right corresponding labels

    """
    if os.path.isfile(filepath):
        with open(filepath) as jfile:
            contents = json.load(jfile)
        labels, pairs = contents['labels'], contents['pairs']
    else:
        labels, pairs = [], []
    return labels, pairs
