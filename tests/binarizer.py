#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from time import time
from memory_profiler import profile

from network_utils.data import Data3d, Transforming3d
from network_utils.transformers import LabelImageBinarizer

image_path = 'data/at1000_image.nii.gz'
label_path = 'data/at1000_label.nii.gz'
mask_path = 'data/at1000_mask.nii.gz'

@profile
def test(on_the_fly=True):
    """Test label image binarize"""
    print('On the fly:', on_the_fly)
    image = Data3d(image_path, on_the_fly=on_the_fly)
    label = Data3d(label_path, on_the_fly=on_the_fly)
    binarizer = LabelImageBinarizer()
    binarizer.update()
    blabel = Transforming3d(label, binarizer, on_the_fly=on_the_fly)

    start_time = time()
    image.get_data()
    blabel.get_data()
    end_time = time()
    print('First binarization', end_time - start_time)

    start_time = time()
    image.get_data()
    blabel.get_data()
    end_time = time()
    print('second binarization', end_time - start_time)

    return image, blabel

if __name__ == "__main__":
    test(on_the_fly=True)
    image, blabel = test(on_the_fly=False)

    plt.figure(figsize=(10, 6))
    print(image.get_data().shape)
    print(blabel.get_data().shape)
    shape = image.get_data().shape[1:]
    channels = [0, 1, 8, 16, 24]
    a_slices = [50, 50, 55, 35, 35]
    c_slices = [145, 145, 160, 166, 150]
    s_slices = [90, 100, 90, 60, 97]
    num_cols = len(channels)
    for i, (channel, a_slice, c_slice, s_slice) in \
            enumerate(zip(channels, a_slices, c_slices, s_slices)):
        plt.subplot(3, num_cols, i + 1)
        plt.imshow(image.get_data()[0, :, :, a_slice], cmap='gray')
        plt.imshow(blabel.get_data()[channel, :, :, a_slice], alpha=0.3)
        plt.subplot(3, num_cols, i + num_cols + 1)
        plt.imshow(image.get_data()[0, :, c_slice, :], cmap='gray')
        plt.imshow(blabel.get_data()[channel, :, c_slice, :], alpha=0.3)
        plt.subplot(3, num_cols, i + num_cols * 2 + 1)
        plt.imshow(image.get_data()[0, s_slice, :, :], cmap='gray')
        plt.imshow(blabel.get_data()[channel, s_slice, :, :], alpha=0.3)
    plt.show()
