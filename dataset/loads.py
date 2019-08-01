# -*- coding: utf-8 -*-

"""Functions loading data from hard drive to memory

"""
import os
import json

from .trees import Leaf, Tree


def load(filepath, dtype):
    """Wrapper

    """
    if filepath.endswith('.npy'):
        return _load_npy(filepath, dtype)
    elif filepath.endswith('.nii') or filepath.endswith('.nii.gz'):
        return _load_nii(filepath, dtype)


def _load_npy(filepath, dtype):
    """Load .npy file
    
    Args:
        filepath (str): The path to the file to load
        dtype (type): The data type of the loaded data

    Returns:
        data (numpy.array): Loaded data

    """
    import numpy as np
    return np.load(filepath).astype(dtype)


def _load_nii(filepath, dtype):
    """Load .nii/.nii.gz file

    Args:
        filepath (str): The path to the file to load
        dtype (type): The data type of the loaded data

    Returns:
        data (numpy.array): Loaded data

    """
    import nibabel as nib
    return nib.load(filepath).get_data().astype(dtype)


def load_tree(filepath):
    if os.path.isfile(filepath):
        with open(filepath) as jfile:
            tree = json.load(jfile)
        return _create_tree(tree)

def _create_tree(tree, level=0):
    if 'subregions' in tree:
        subtrees = {sub['name']: _create_tree(sub, level=level+1)
                    for sub in tree['subregions']}
        return Tree(subtrees, level=level)
    else:
        return Leaf(level=level)
