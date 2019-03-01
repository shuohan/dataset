#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from time import time
from memory_profiler import profile

from network_utils.workers import Cropper
from network_utils.images import Image, Label, Mask


image_path = 'data/at1000_image.nii.gz'
label_path = 'data/at1000_label.nii.gz'
mask_path = 'data/at1000_mask.nii.gz'

def test(cropping_shape=(128, 96, 96), on_the_fly=True):
    """Test Cropper"""
    print('Cropping shape:', cropping_shape)
    image = Image(filepath=image_path, on_the_fly=on_the_fly)
    label = Label(filepath=label_path, on_the_fly=on_the_fly)
    mask = Mask(filepath=mask_path, on_the_fly=on_the_fly)
    cropper = Cropper()
    image, label = cropper.process(image, label, mask)

    return image, label

if __name__ == "__main__":
    test(on_the_fly=True)
    image, label = test(on_the_fly=False)

    plt.figure()
    shape = image.shape
    print('Cropped shape', shape)
    plt.subplot(1, 3, 1)
    plt.imshow(image.data[shape[0]//2, :, :], cmap='gray')
    plt.imshow(label.data[shape[0]//2, :, :], cmap='tab20', alpha=0.3)
    plt.subplot(1, 3, 2)
    plt.imshow(image.data[:, shape[1]//2, :], cmap='gray')
    plt.imshow(label.data[:, shape[1]//2, :], cmap='tab20', alpha=0.3)
    plt.subplot(1, 3, 3)
    plt.imshow(image.data[:, :, shape[2]//2], cmap='gray')
    plt.imshow(label.data[:, :, shape[2]//2], cmap='tab20', alpha=0.3)
    plt.show()
