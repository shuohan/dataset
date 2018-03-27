#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt

from network_utils import MedicalImageCropSegDataset3d

dataset = MedicalImageCropSegDataset3d('data')

for image, label, mask in dataset:

    plt.figure()

    plt.subplot(3, 3, 1)
    sliceid = image.shape[0] // 2
    plt.imshow(image[sliceid, :, :], cmap='gray')
    plt.subplot(3, 3, 2)
    sliceid = image.shape[1] // 2
    plt.imshow(image[:, sliceid, :], cmap='gray')
    plt.subplot(3, 3, 3)
    sliceid = image.shape[2] // 2
    plt.imshow(image[:, :, sliceid], cmap='gray')

    plt.subplot(3, 3, 4)
    sliceid = image.shape[0] // 2
    plt.imshow(mask[sliceid, :, :], cmap='gray')
    plt.subplot(3, 3, 5)
    sliceid = image.shape[1] // 2
    plt.imshow(mask[:, sliceid, :], cmap='gray')
    plt.subplot(3, 3, 6)
    sliceid = image.shape[2] // 2
    plt.imshow(mask[:, :, sliceid], cmap='gray')

    plt.subplot(3, 3, 7)
    sliceid = image.shape[0] // 2
    plt.imshow(label[sliceid, :, :])
    plt.subplot(3, 3, 8)
    sliceid = image.shape[1] // 2
    plt.imshow(label[:, sliceid, :])
    plt.subplot(3, 3, 9)
    sliceid = image.shape[2] // 2
    plt.imshow(label[:, :, sliceid])

plt.show()
