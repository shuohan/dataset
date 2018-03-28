# -*- coding: utf-8 -*-

import os
from glob import glob


class Dataset3d:
    """Dataset of 3D or 4D (multi-channel 3D) images

    Attributes:
        
    """
    def __init__(self, data):
        """Default initialize

        Args:
            data (list of list of data.Data3d): The data to handle

        """
        assert all([len(d) == len(data[0]) for d in data[1:]])
        self.data = data

    @classmethod
    def from_directory(cls, dirname, patterns, **kwargs):
        """Initialize from directory

        Args:
            dirname (str): The the name of the directory containing data
            patterns (list of str): The filename patterns of the data to load
            kwargs (dict): The settings to pass to data.Data3d

        """
        data = list()
        for p in patterns:
            filepaths = _get_filepaths_with_pattern(dirname, p)
            data.append([Data3d(fp, **kwargs) for fp in filepaths])
        return cls.__init__(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return [data[index] for data in self.data]


    def combine(self, dataset):
        """Combine with other dataset

        Args:
            dataset (Dataset3d): The dataset to combine

        """
        combined_data = [data + other_data
                         for data, other_data in zip(self.data, dataset.data)]
        return self.__init__(combined_data)

    def split(self, indices):
        """Split self to two datasets

        Args:
            indices (list of int): The indices in the original data of the first
                split dataset

        """
        indices_all = set(range(len(self)))
        indices1 = set(indices)
        indices2 = indices_all - indices1
        data1 = [self.data[i] for i in sorted(list(indices1))]
        data2 = [self.data[i] for i in sorted(list(indices2))]
        return self.__init__(data1), self.__init__(data2)
        

def _get_filepaths_with_pattern(self, dirname, pattern):
    return sorted(glob(os.path.join(dirname, pattern)))
