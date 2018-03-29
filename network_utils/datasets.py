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
            data (list of list of data.Data): The data to handle

        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        self.data[index][0].update()
        return [d.get_data() for d in self.data[index]]

    def __add__(self, dataset):
        """Combine with other dataset

        Args:
            dataset (Dataset3d): The dataset to combine
        
        Returns:
            combined_dataset (Dataset3d): The combined datasets

        """
        combined_data = self.data + dataset.data
        return Dataset3d(combined_data)

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
        return Dataset3d(data1), Dataset3d(data2)
