# -*- coding: utf-8 -*-

"""Functions loading data from hard drive to memory

"""
import os
import json


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
        labels, pairs = dict(), list()
    # labels = {int(k): v for k, v in labels.items()}
    return labels, pairs


#TODO
def load_label_hierachy(filepath):
    if os.path.isfile(filepath):
        with open(filepath) as jfile:
            contents = json.load(jfile)
        hierachy = Hierachy.create_child(contents)
    else:
        hierachy = Region(value=None)
    return hierachy


class Region:

    def __init__(self, name):
        self.name = name
        self.print_level = 0

    @property
    def regions(self):
        return [self.name]

    def __str__(self):
        return self._get_space() + '- ' + self.regions[0]

    def _get_space(self):
        return '|   ' * self.print_level


class Hierachy(Region):

    def __init__(self, name, children):
        self.name = name
        self.children = [self.create_child(child) for child in children]
        self.print_level = 0

    @staticmethod
    def create_child(child):
        if 'children' in child:
            return Hierachy(child['name'], child['children'])
        else:
            return Region(child['name'])

    @property
    def regions(self):
        results = list()
        for child in self.children:
            results.extend(child.regions)
        return results

    def __str__(self):
        result = list()
        result.append(self._get_space() + '- ' + self.name)
        for child in self.children:
            child.print_level = self.print_level + 1
            result.append(child.__str__())
        return '\n'.join(result)
