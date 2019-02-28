#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from time import time
from memory_profiler import profile

from network_utils.data import Data3d, Interpolating3d
from network_utils.transformers import Scaler


image_path = 'data/at1000_image.nii.gz'
label_path = 'data/at1000_label.nii.gz'

@profile
def test(max_scale=3, on_the_fly=True):
    """Test Scaler"""
    print('Maximum scale:', max_scale)
    print('On the fly:', on_the_fly)
    image = Data3d(image_path, on_the_fly=on_the_fly)
    label = Data3d(label_path, on_the_fly=on_the_fly)
    scaler = Scaler(max_scale=max_scale)
    simage = Interpolating3d(image, scaler, on_the_fly=on_the_fly, order=1)
    slabel = Interpolating3d(label, scaler, on_the_fly=on_the_fly, order=0)

    scaler.update()
    start_time = time()
    simage.get_data()
    slabel.get_data()
    end_time = time()
    print('First translation', end_time - start_time)
    scaler.cleanup()

    scaler.update()
    start_time = time()
    simage.get_data()
    slabel.get_data()
    end_time = time()
    print('Second translate', end_time - start_time)
    scaler.cleanup()

    return simage, slabel

if __name__ == "__main__":
    simage, slabel = test(on_the_fly=True)
    test(on_the_fly=False)

    sagittal = 90
    axial = 50
    coronal = 150

    simage.update()
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(simage.get_data()[0, sagittal, :, :], cmap='gray')
    plt.imshow(slabel.get_data()[0, sagittal, :, :], cmap='tab20', alpha=0.3)
    plt.subplot(1, 3, 2)
    plt.imshow(simage.get_data()[0, :, coronal, :], cmap='gray')
    plt.imshow(slabel.get_data()[0, :, coronal, :], cmap='tab20', alpha=0.3)
    plt.subplot(1, 3, 3)
    plt.imshow(simage.get_data()[0, :, :, axial], cmap='gray')
    plt.imshow(slabel.get_data()[0, :, :, axial], cmap='tab20', alpha=0.3)
    simage.cleanup()

    simage.update()
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(simage.get_data()[0, sagittal, :, :], cmap='gray')
    plt.imshow(slabel.get_data()[0, sagittal, :, :], cmap='tab20', alpha=0.3)
    plt.subplot(1, 3, 2)
    plt.imshow(simage.get_data()[0, :, coronal, :], cmap='gray')
    plt.imshow(slabel.get_data()[0, :, coronal, :], cmap='tab20', alpha=0.3)
    plt.subplot(1, 3, 3)
    plt.imshow(simage.get_data()[0, :, :, axial], cmap='gray')
    plt.imshow(slabel.get_data()[0, :, :, axial], cmap='tab20', alpha=0.3)
    simage.cleanup()
    plt.show()
