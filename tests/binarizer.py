#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')

from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from time import time

from network_utils.label_image_binarizer import LabelImageBinarizer
from network_utils.data_factory import TrainingDataFactory, Data3dFactoryCropper
from network_utils.data_factory import Data3dFactoryBinarizer
from network_utils.datasets import Dataset3d


image_paths = sorted(glob('data/*image.nii.gz'))
label_paths = sorted(glob('data/*label.nii.gz'))
mask_paths = sorted(glob('data/*mask.nii.gz'))

types = ['none', 'rotation']
label_pairs = [[33, 36], [43, 46], [53, 56], [63, 66], [73, 76], [74, 77],
               [75, 78], [83, 86], [84, 87], [93, 96], [103, 106]]
factory = TrainingDataFactory(dim=1, label_pairs=label_pairs, max_angle=20,
                              get_data_on_the_fly=False, types=types)
factory = Data3dFactoryCropper(factory, (128, 96, 96))

binarizer = LabelImageBinarizer()
factory = Data3dFactoryBinarizer(factory, binarizer)

data = list()
counter = 0
for ip, lp, mp in zip(image_paths, label_paths, mask_paths):
    print(ip, lp, mp)
    factory.create_data(ip, lp, mp)
    data.append(factory.data)
    counter += 1
    if counter == 2:
        break

data= {key:[d[key] for d in data] for key in data[0].keys()}
num_repeats = 2

for k, v in data.items():
    print(k)
    dataset = Dataset3d(v)

    for i in range(num_repeats):

        for image, label in dataset:
            image = image[0, ...]

            plt.figure()

            plt.subplot(2, 3, 1)
            sliceid = image.shape[0] // 2
            plt.imshow(image[sliceid, :, :], cmap='gray', alpha=0.7)
            plt.imshow(label[0, sliceid, :, :], alpha=0.3)
            plt.subplot(2, 3, 2)
            sliceid = image.shape[1] // 2
            plt.imshow(image[:, sliceid, :], cmap='gray', alpha=0.7)
            plt.imshow(label[0, :, sliceid, :], alpha=0.3)
            plt.subplot(2, 3, 3)
            sliceid = image.shape[2] // 2
            plt.imshow(image[:, :, sliceid], cmap='gray', alpha=0.7)
            plt.imshow(label[0, :, :, sliceid], alpha=0.3)

            plt.subplot(2, 3, 4)
            sliceid = image.shape[0] // 2
            plt.imshow(image[sliceid, :, :], cmap='gray', alpha=0.7)
            plt.imshow(label[1, sliceid, :, :], alpha=0.3)
            plt.subplot(2, 3, 5)
            sliceid = image.shape[1] // 2
            plt.imshow(image[:, sliceid, :], cmap='gray', alpha=0.7)
            plt.imshow(label[1, :, sliceid, :], alpha=0.3)
            plt.subplot(2, 3, 6)
            sliceid = image.shape[2] // 2
            plt.imshow(image[:, :, sliceid], cmap='gray', alpha=0.7)
            plt.imshow(label[1, :, :, sliceid], alpha=0.3)

plt.show()
