#!/usr/bin/env python
# -*- coding: utf-8 -*-


import sys
sys.path.insert(0, '..')

from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import os
from memory_profiler import profile

from network_utils.datasets import Dataset3dFactory
from network_utils.data_factories import TrainingDataFactory


load_on_the_fly = True

# types = ['none', 'flipping', 'rotation', 'deformation', 'translation']
types = ['none', 'scaling']
label_pairs = [[33, 36], [43, 46], [53, 56], [63, 66], [73, 76], [74, 77],
               [75, 78], [83, 86], [84, 87], [93, 96], [103, 106]]
data_factory = TrainingDataFactory(dim=1, label_pairs=label_pairs,
                                   max_trans=20, max_angle=20,
                                   get_data_on_the_fly=load_on_the_fly,
                                   types=types)

image_paths = sorted(glob('data/*image.nii.gz'))[:2]
label_paths = sorted(glob('data/*label.nii.gz'))[:2]
mask_paths = sorted(glob('data/*mask.nii.gz'))[:2]

# t_dataset, v_dataset = Dataset3dFactory.create(data_factory, [],
#                                                image_paths, label_paths,
#                                                mask_paths=mask_paths,
#                                                cropping_shape=(128,96,96))
t_dataset, v_dataset = Dataset3dFactory.create(data_factory, [],
                                               image_paths, label_paths)

@profile
def get():
    for i, (image, label) in enumerate(t_dataset):
        image = image[0, ...]
        label = label[0, ...]
        name = os.path.basename(t_dataset.data[i][0].filepath).split('.')[0]
        print(name)

        plt.figure()

        plt.subplot(1, 3, 1)
        sliceid = image.shape[0] // 2
        plt.imshow(image[sliceid, :, :], cmap='gray', alpha=0.7)
        plt.imshow(label[sliceid, :, :], alpha=0.3)
        plt.subplot(1, 3, 2)
        sliceid = image.shape[1] // 2
        plt.imshow(image[:, sliceid, :], cmap='gray', alpha=0.7)
        plt.imshow(label[:, sliceid, :], alpha=0.3)
        plt.subplot(1, 3, 3)
        sliceid = image.shape[2] // 2
        plt.imshow(image[:, :, sliceid], cmap='gray', alpha=0.7)
        plt.imshow(label[:, :, sliceid], alpha=0.3)
        plt.title(name)

    for i, (image, label) in enumerate(v_dataset):
        image = image[0, ...]
        label = label[0, ...]
        name = os.path.basename(t_dataset.data[i][0].filepath).split('.')[0]
        print(name)

        plt.figure()

        plt.subplot(1, 3, 1)
        sliceid = image.shape[0] // 2
        plt.imshow(image[sliceid, :, :], cmap='gray', alpha=0.7)
        plt.imshow(label[sliceid, :, :], alpha=0.3)
        plt.subplot(1, 3, 2)
        sliceid = image.shape[1] // 2
        plt.imshow(image[:, sliceid, :], cmap='gray', alpha=0.7)
        plt.imshow(label[:, sliceid, :], alpha=0.3)
        plt.subplot(1, 3, 3)
        sliceid = image.shape[2] // 2
        plt.imshow(image[:, :, sliceid], cmap='gray', alpha=0.7)
        plt.imshow(label[:, :, sliceid], alpha=0.3)
        plt.title(name)

    plt.show()

get()
