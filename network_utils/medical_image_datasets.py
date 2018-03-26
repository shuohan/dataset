# -*- coding: utf-8 -*-

import os
from glob import glob
from torch.utils.data import Dataset


class MedicalImageDataset3d(Dataset):
    """Dataset containing medical images

    Only handle 3D images

    Attributes:
        
    """
    source_pattern = '*source*'
    target_pattern = '*target*'

    def __init__(self, dirname, **kwargs):
        self.dirname = dirname
        source_filepaths = self._get_filepaths_with_pattern(self.source_pattern)
        target_filepaths = self._get_filepaths_with_pattern(self.target_pattern)
        assert len(source_filepaths) == len(target_filepaths)
        self.sources = [MedicalImage3d(f, **kwargs) for f in source_filepaths]
        self.targets = [MedicalImage3d(f, **kwargs) for f in target_filepaths]

    def __getitem__(self, index):
        return self.sources[index].data, self.targets[index].data

    def __len__(self):
        return len(self.sources)

    def _get_filepaths_with_pattern(self, pattern):
        return sorted(glob(os.path.join(self.dirname, pattern)))


class MedicalImageSegDataset3d(MedicalImageDataset3d):
    source_pattern = '*image*'
    target_pattern = '*label*'


class MedicalImageCropSegDataset3d(MedicalImageSegDataset3d):
    mask_pattern = '*mask*'
    def __init__(self, dirname, **kwargs):
        super().__init__(dirname, **kwargs)
        mask_filepaths = self._get_filepaths_with_pattern(self.mask_pattern)
        assert len(source_filepaths) == len(mask_filepaths)
        self.masks = [MedicalImage3d(f, **kwargs) for f in mask_filepaths]

    def __getitem__(self, index):
        source = self.sources[index].data
        target = self.targets[index].data
        mask = self.masks[index].data
        return source, target, mask


class MedicalImage3d:
    """Object handling a 3D medical image

    """
    def __init__(self, filepath, load_on_the_fly=True, transpose4d=True):

        self.filepath = filepath
        self.load_on_the_fly = load_on_the_fly
        self.transpose4d = transpose4d

        if filepath.endswith('.npy'):
            self._load_func = load_npy
        elif filepath.endswith('.nii') of filepath.endswith('.nii.gz'):
            self._load_func = load_nii

        self._data = None

    @property
    def data(self):
        """Access the loaded data
        
        Returns:
            data (numpy.array): Loaded data

        """
        if self.load_on_the_fly:
            return self._load()
        else:
            if self._data is None:
                self._data = self._load()
            return self._data

    def _load(self):
        """Load data from hard drive. Need to override

        Returns:
            data (num_channels x num_i ... numpy.array): The loaded data

        """
        data = self._load_func(self.filepath)
        if len(data.shape) == 4:
            if self._transpose:
                data = np.transpose(data, [3, 0, 1, 2])
        else:
            data = data[None, ...]
        return data


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
