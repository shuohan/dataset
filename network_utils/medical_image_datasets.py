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
        assert len(self.sources) == len(mask_filepaths)
        self.masks = [MedicalImage3d(f, **kwargs) for f in mask_filepaths]

    def __getitem__(self, index):
        source = self.sources[index].data
        target = self.targets[index].data
        mask = self.masks[index].data
        return source, target, mask


class DummyDataset(Dataset):

    def __init__(self, sources, targets):
        assert len(sources) == len(targets)
        self.sources = sources
        self.targets = targets

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, index):
        return self.sources[index].data, self.targets[index].data


class DummyDatasetCrop(Dataset):

    def __init__(self, sources, targets, masks):
        assert len(sources) == len(targets)
        assert len(sources) == len(masks)
        self.sources = sources
        self.targets = targets
        self.masks = masks

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, index):
        source = self.sources[index].data
        target = self.targets[index].data
        mask = self.masks[index].data
        return source, target, mask


def split_dataset(dataset, ids):
    all_indices = set(range(len(dataset)))
    other_indices = all_indices - set(ids)
    ids1 = sorted(list(ids))
    ids2 = sorted(list(other_indices))
    sources1 = [dataset.sources[id] for id in ids1]
    targets1 = [dataset.targets[id] for id in ids1]
    sources2 = [dataset.sources[id] for id in ids2]
    targets2 = [dataset.targets[id] for id in ids2]
    dummy1 = DummyDataset(sources1, targets1)
    dummy2 = DummyDataset(sources2, targets2)
    return dummy1, dummy2


def split_dataset_crop(dataset, ids):
    all_indices = set(range(len(dataset)))
    other_indices = all_indices - set(ids)
    ids1 = sorted(list(ids))
    ids2 = sorted(list(other_indices))
    sources1 = [dataset.sources[id] for id in ids1]
    targets1 = [dataset.targets[id] for id in ids1]
    masks1 = [dataset.masks[id] for id in ids1]
    sources2 = [dataset.sources[id] for id in ids2]
    targets2 = [dataset.targets[id] for id in ids2]
    masks2 = [dataset.masks[id] for id in ids2]
    dummy1 = DummyDatasetCrop(sources1, targets1, masks1)
    dummy2 = DummyDatasetCrop(sources2, targets2, masks2)
    return dummy1, dummy2


class MedicalImage3d:
    """Object handling a 3D medical image

    """
    def __init__(self, filepath, load_on_the_fly=True, transpose4d=True):

        self.filepath = filepath
        self.load_on_the_fly = load_on_the_fly
        self.transpose4d = transpose4d

        if filepath.endswith('.npy'):
            self._load_func = load_npy
        elif filepath.endswith('.nii') or filepath.endswith('.nii.gz'):
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
            # data = data[None, ...]
            data = data
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
