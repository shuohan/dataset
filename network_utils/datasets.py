# -*- coding: utf-8 -*-


class Dataset:
    """Implements __len__ and __getiem__

    Attributes:
        data (list of tuple of .data.Data or .data.DataDecorator): Data holder.
            The tuple should contain the image/label image pair etc. and the
            transfomered applied to them should be the same so update/cleanup
            one from the tuple should also update/clean the rest of the tuple
        
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """Get an item from the dataset

        Args:
            key (int): The index

        Returns:
            results (tuple of numpy.array): The data to retrieve

        """
        assert type(key) is int
        data = self.data[key] # (image, image_image, ect.)
        data[0].update() # update the first should also update the rest
        results = tuple([d.get_data() for d in data])
        data[0].cleanup()# cleanup the first should also cleanup the rest
        return results

    def __add__(self, dataset):
        """Combine with other dataset

        Args:
            dataset (Dataset3d): The dataset to combine
        
        Returns:
            combined_dataset (Dataset3d): The combined datasets

        """
        combined_data = self.data + dataset.data
        return Dataset(combined_data)

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
