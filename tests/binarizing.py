#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt

from network_utils import MedicalImageCropSegDataset3d
from network_utils import CroppedMedicalImageDataset3d
from network_utils import BinarizedMedicalImageDataset3d
from network_utils import LabelImageBinarizer

binarizer = LabelImageBinarizer()
dataset = MedicalImageCropSegDataset3d('data')
dataset = CroppedMedicalImageDataset3d(dataset, (128, 96, 96))
dataset = BinarizedMedicalImageDataset3d(dataset, binarizer)

for image, label in dataset:

    plt.figure()

    plt.subplot(1, 3, 1)
    sliceid = image.shape[0] // 2
    plt.imshow(image[sliceid, :, :], cmap='gray')
    plt.subplot(1, 3, 2)
    sliceid = image.shape[1] // 2
    plt.imshow(image[:, sliceid, :], cmap='gray')
    plt.subplot(1, 3, 3)
    sliceid = image.shape[2] // 2
    plt.imshow(image[:, :, sliceid], cmap='gray')
    plt.figure()

    for i, channel in enumerate(label):
        plt.subplot(4, 8, i + 1)
        sliceid = image.shape[1] // 2
        plt.imshow(channel[:, sliceid, :])

    break

plt.show()
