#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt

from network_utils import MedicalImageCropSegDataset3d, split_dataset_crop
from network_utils import CroppedMedicalImageDataset3d

dataset = MedicalImageCropSegDataset3d('data')
dataset1, dataset2 = split_dataset_crop(dataset, [2, 6])
dataset1 = CroppedMedicalImageDataset3d(dataset1, (128, 96, 96))
dataset2 = CroppedMedicalImageDataset3d(dataset2, (128, 96, 96))

print(len(dataset), len(dataset1), len(dataset2))

for image, label in dataset1:

    plt.figure(figsize=(3, 1.5))

    sliceid = image.shape[1] // 2
    plt.subplot(1, 2, 1)
    plt.imshow(image[:, sliceid, :], cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(label[:, sliceid, :])

for image, label in dataset2:

    plt.figure(figsize=(3, 1.5))

    sliceid = image.shape[1] // 2
    plt.subplot(1, 2, 1)
    plt.imshow(image[:, sliceid, :], cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(label[:, sliceid, :])

plt.show()
