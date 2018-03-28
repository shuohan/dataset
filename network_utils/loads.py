# -*- coding: utf-8 -*-
"""Functions loading data from hard drive to memory

"""

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
