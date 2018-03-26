#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from network_utils import MedicalImageCropSegDataset3d
from network_utils import CroppedMedicalImageDataset3d
from network_utils import TransformedMedicalImageDataset3d
from network_utils import random_rotate3d

from torchvision.transforms import Compose

rotate = partial(random_rotate3d, max_angle=20)

dataset = MedicalImageCropSegDataset3d('data')
t_dataset = TransformedMedicalImageDataset3d(dataset, rotate)
dataset = CroppedMedicalImageDataset3d(dataset, (128, 96, 96))
t_dataset = CroppedMedicalImageDataset3d(t_dataset, (128, 96, 96))

for (image, label), (t_image, t_label) in zip(dataset, t_dataset):
    plt.figure()

    plt.subplot(2, 6, 1)
    sliceid = image.shape[0] // 2
    plt.imshow(image[sliceid, :, :], cmap='gray')
    plt.subplot(2, 6, 2)
    sliceid = image.shape[1] // 2
    plt.imshow(image[:, sliceid, :], cmap='gray')
    plt.subplot(2, 6, 3)
    sliceid = image.shape[2] // 2
    plt.imshow(image[:, :, sliceid], cmap='gray')

    plt.subplot(2, 6, 4)
    sliceid = image.shape[0] // 2
    plt.imshow(label[sliceid, :, :])
    plt.subplot(2, 6, 5)
    sliceid = image.shape[1] // 2
    plt.imshow(label[:, sliceid, :])
    plt.subplot(2, 6, 6)
    sliceid = image.shape[2] // 2
    plt.imshow(label[:, :, sliceid])

    plt.subplot(2, 6, 7)
    sliceid = image.shape[0] // 2
    plt.imshow(t_image[sliceid, :, :], cmap='gray')
    plt.subplot(2, 6, 8)
    sliceid = image.shape[1] // 2
    plt.imshow(t_image[:, sliceid, :], cmap='gray')
    plt.subplot(2, 6, 9)
    sliceid = image.shape[2] // 2
    plt.imshow(t_image[:, :, sliceid], cmap='gray')

    plt.subplot(2, 6, 10)
    sliceid = image.shape[0] // 2
    plt.imshow(t_label[sliceid, :, :])
    plt.subplot(2, 6, 11)
    sliceid = image.shape[1] // 2
    plt.imshow(t_label[:, sliceid, :])
    plt.subplot(2, 6, 12)
    sliceid = image.shape[2] // 2
    plt.imshow(t_label[:, :, sliceid])

    break

plt.show()
