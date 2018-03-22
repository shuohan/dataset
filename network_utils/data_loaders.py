# -*- coding: utf-8 -*-

"""Data loaders

Load data from hard drive

"""

import numpy as np


class DataLoader:
    """Load data from hard drive. Abstract class

    Attributes:
        filename (str): The path to the file to load
        load_on_the_fly (bool): If True, reload the data each time `self.data`
            is accessed. Otherwise, only load the data at the first access and
            keep track of it
        data (numpy array): Access the loaded data
        _load_func (function): The function used to load the file
        _data (numpy array): Loaded data. None means it has not been loaded or
            it is not kept track

    """
    def __init__(self, filename, load_on_the_fly=False, load_func=load_npy):
        """Initilize

        Args:
            filename (str): The path to the file to load
        load_on_the_fly (bool): If True, reload the data each time `self.data`
            is accessed. Otherwise, only load the data at the first access and
            keep track of it
        load_func (function): The function used to load the file

        """
        self.filename = filename
        self.load_on_the_fly = load_on_the_fly
        self._load_func = load_func
        self._data = None

    @property
    def data(self):
        """Access the loaded data
        
        Returns:
            data (numpy array): Loaded data

        """
        if self.load_on_the_fly:
            return self._load()
        else:
            if self._data is None:
                self._data = self._load()
            return self._data

    def _load(self):
        """Load data from hard drive. Need to override

        """
        self._load_func(self.filename)
        raise NotImplementedError


class ImageLoader(DataLoader):
    """Load image from hard drive
    
    Attribute:
        data (num_channels x dim_i x dim_j x dim_k numpy array): The loaded
            image. Assume a single channel image

    """
    def _load(self):
        """Load single channel image

        Returns:
            data (num_channels x dim_i x dim_j x dim_k numpy array): Loaded
                single channel image

        """
        return self._load_func(self.filename)[None, ...]

class MaskLoader(DataLoader):
    """Load mask from hard drive

    Attributes:
        data (dim_i x dim_j x dim_k bool numpy image): Loaded 3D mask

    """
    def _load(self):
        """Load mask from the hard drive

        Returns:
            data (dim_i x dim_j x dim_k bool numpy image): Loaded 3D mask

        """
        return self._load_func(self.filename).astype(bool)


class LabelImageLoader(DataLoader):
    """Load label image from the hard drive

    The label image has to be binarized. For example, if the label image is
    [1, 12; 3, 12] with 3 different labels. Binarization first map these labels
    to binary codes 1 -> [1, 0, 0], 3 -> [0, 1, 0], 12 -> [0, 0, 1], and the
    label image is converted to [1, 0; 0, 0], [0, 0; 1, 0], and [0, 1; 0, 1]
    for these 3 binary channels. This is essential to network softmax output.
    If there is only 2 labels, the result num_channels is 1

    Attributes:
        data (num_channels x dim_i x dim_j x dim_k numpy array): The loaded
            label image

    """
    def _load(self):
        """ Load label image

        Returns:
            label_image (num_channels x dim_i x dim_j x dim_k numpy array): The
                loaded label image. One channel if their are two labels;
                Multiple channels if more than two labels

        """
        return self._load_func(self.filename).astype(int)[None, ...]


def load_npy(filepath):
    """Load .npy file
    
    Args:
        filepath (str): The path to the file to load

    Returns:
        data (numpy.array): Loaded data

    """
    return np.load(filepath)


def load_nii(filepath):
    """Load .nii/.nii.gz file

    Args:
        filepath (str): The path to the file to load

    Returns:
        data (numpy.array): Loaded data

    """
    import nibabel as nib
    return nib.load(filepath).get_data()


def load_to_pytorch_variable(filepath, load_func=load_nii, use_gpu=True,
                             dtype='float'):
    """Load to PyTorch Valiable

    Args:
        filepath (str): The path to the file to load
        load_func (function): The loading function
        use_gpu (bool): Load the data to GPU
        dtype (str): The data type
    
    Returns:
        data (torch.nn.Variable): The loaded data

    """
    import torch
    if dtype == 'float':
        dtype = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
    if dtype == 'byte':
        dtype = torch.cuda.ByteTensor if use_gpu else torch.ByteTensor
    npy = load_func(filepath)
    data = torch.autograd.Variable(dtype(npy))
    return data
