#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt

from network_utils import MedicalImageCropSegDataset3d
from network_utils import CroppedMedicalImageDataset3d

dataset = MedicalImageCropSegDataset3d('data')
dataset = CroppedMedicalImageDataset3d(dataset, (128, 96, 96))

for image, label in dataset:

    plt.figure()

    plt.subplot(2, 3, 1)
    sliceid = image.shape[0] // 2
    plt.imshow(image[sliceid, :, :], cmap='gray')
    plt.subplot(2, 3, 2)
    sliceid = image.shape[1] // 2
    plt.imshow(image[:, sliceid, :], cmap='gray')
    plt.subplot(2, 3, 3)
    sliceid = image.shape[2] // 2
    plt.imshow(image[:, :, sliceid], cmap='gray')

    plt.subplot(2, 3, 4)
    sliceid = image.shape[0] // 2
    plt.imshow(label[sliceid, :, :])
    plt.subplot(2, 3, 5)
    sliceid = image.shape[1] // 2
    plt.imshow(label[:, sliceid, :])
    plt.subplot(2, 3, 6)
    sliceid = image.shape[2] // 2
    plt.imshow(label[:, :, sliceid])

plt.show()
