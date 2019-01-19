#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from time import time
from memory_profiler import profile

from network_utils.data import Data3d, Transforming3d
from network_utils.transformers import Cropper


image_path = 'data/at1000_image.nii.gz'
label_path = 'data/at1000_label.nii.gz'
mask_path = 'data/at1000_mask.nii.gz'

@profile
def test(cropping_shape=(128, 96, 96), on_the_fly=True):
    """Test Cropper"""
    print('Cropping shape:', cropping_shape)
    print('On the fly:', on_the_fly)
    image = Data3d(image_path, on_the_fly=on_the_fly)
    label = Data3d(label_path, on_the_fly=on_the_fly)
    mask = Data3d(mask_path, on_the_fly=on_the_fly)
    cropper = Cropper(mask, cropping_shape)
    cimage = Transforming3d(image, cropper, on_the_fly=on_the_fly)
    clabel = Transforming3d(label, cropper, on_the_fly=on_the_fly)

    cropper.update()
    start_time = time()
    cimage.get_data()
    clabel.get_data()
    end_time = time()
    print('First crop', end_time - start_time)
    cropper.cleanup()

    cropper.update()
    start_time = time()
    cimage.get_data()
    clabel.get_data()
    end_time = time()
    print('second load', end_time - start_time)
    cropper.cleanup()

    return cimage, clabel

if __name__ == "__main__":
    test(on_the_fly=True)
    cimage, clabel = test(on_the_fly=False)

    plt.figure()
    shape = cimage.shape[1:]
    print('Cropped shape', shape)
    plt.subplot(1, 3, 1)
    plt.imshow(cimage.get_data()[0, shape[0]//2, :, :], cmap='gray')
    plt.imshow(clabel.get_data()[0, shape[0]//2, :, :], cmap='tab20', alpha=0.3)
    plt.subplot(1, 3, 2)
    plt.imshow(cimage.get_data()[0, :, shape[1]//2, :], cmap='gray')
    plt.imshow(clabel.get_data()[0, :, shape[1]//2, :], cmap='tab20', alpha=0.3)
    plt.subplot(1, 3, 3)
    plt.imshow(cimage.get_data()[0, :, :, shape[2]//2], cmap='gray')
    plt.imshow(clabel.get_data()[0, :, :, shape[2]//2], cmap='tab20', alpha=0.3)
    plt.show()
