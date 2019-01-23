#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from time import time
from memory_profiler import profile

from network_utils.data import Image3d, Label3d, Interpolating3d
from network_utils.transformers import Rotater


image_path = 'data/at1000_image.nii.gz'
label_path = 'data/at1000_label.nii.gz'

@profile
def test(max_angle=20, on_the_fly=True):
    """Test Rotater"""
    print('Maximum angle:', max_angle)
    print('On the fly:', on_the_fly)
    image = Image3d(image_path, on_the_fly=on_the_fly)
    label = Label3d(label_path, on_the_fly=on_the_fly)
    rotator = Rotater(max_angle=max_angle, point=None)
    rimage = Interpolating3d(image, rotator, on_the_fly=False)
    rlabel = Interpolating3d(label, rotator, on_the_fly=False)

    rotator.update()
    start_time = time()
    rimage.get_data()
    rlabel.get_data()
    end_time = time()
    print('First rotate', end_time - start_time)
    rotator.cleanup()

    rotator.update()
    start_time = time()
    rimage.get_data()
    rlabel.get_data()
    end_time = time()
    print('Second rotate', end_time - start_time)
    rotator.cleanup()

    return rimage, rlabel

if __name__ == "__main__":
    rimage, rlabel = test(on_the_fly=True)
    test(on_the_fly=False)

    sagittal = 90
    axial = 50
    coronal = 150

    rimage.update()
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(rimage.get_data()[0, sagittal, :, :], cmap='gray')
    plt.imshow(rlabel.get_data()[0, sagittal, :, :], cmap='tab20', alpha=0.3)
    plt.subplot(1, 3, 2)
    plt.imshow(rimage.get_data()[0, :, coronal, :], cmap='gray')
    plt.imshow(rlabel.get_data()[0, :, coronal, :], cmap='tab20', alpha=0.3)
    plt.subplot(1, 3, 3)
    plt.imshow(rimage.get_data()[0, :, :, axial], cmap='gray')
    plt.imshow(rlabel.get_data()[0, :, :, axial], cmap='tab20', alpha=0.3)
    rimage.cleanup()

    rimage.update()
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(rimage.get_data()[0, sagittal, :, :], cmap='gray')
    plt.imshow(rlabel.get_data()[0, sagittal, :, :], cmap='tab20', alpha=0.3)
    plt.subplot(1, 3, 2)
    plt.imshow(rimage.get_data()[0, :, coronal, :], cmap='gray')
    plt.imshow(rlabel.get_data()[0, :, coronal, :], cmap='tab20', alpha=0.3)
    plt.subplot(1, 3, 3)
    plt.imshow(rimage.get_data()[0, :, :, axial], cmap='gray')
    plt.imshow(rlabel.get_data()[0, :, :, axial], cmap='tab20', alpha=0.3)
    rimage.cleanup()
    plt.show()
