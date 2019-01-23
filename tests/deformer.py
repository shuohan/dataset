#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from time import time
from scipy.ndimage.measurements import center_of_mass
from memory_profiler import profile

from network_utils.data import Image3d, Label3d, Interpolating3d
from network_utils.transformers import Deformer


image_path = 'data/at1000_image.nii.gz'
label_path = 'data/at1000_label.nii.gz'

@profile
def test(scale=30, sigma=5, on_the_fly=True):
    """Test Deformer"""
    print('Scale:', scale)
    print('Sigma:', sigma)
    print('On the fly:', on_the_fly)
    image = Image3d(image_path, on_the_fly=on_the_fly)
    label = Label3d(label_path, on_the_fly=on_the_fly)
    shape = image.shape
    deformer = Deformer(shape, sigma, scale)
    dimage = Interpolating3d(image, deformer, on_the_fly=on_the_fly)
    dlabel = Interpolating3d(label, deformer, on_the_fly=on_the_fly)

    deformer.update()
    start_time = time()
    dimage.get_data()
    dlabel.get_data()
    end_time = time()
    print('First rotate', end_time - start_time)
    deformer.cleanup()

    deformer.update()
    start_time = time()
    dimage.get_data()
    dlabel.get_data()
    end_time = time()
    print('Second rotate', end_time - start_time)
    deformer.cleanup()

    return dimage, dlabel

if __name__ == "__main__":
    dimage, dlabel = test(on_the_fly=True)
    test(on_the_fly=False)

    sagittal = 90
    axial = 50
    coronal = 150

    dimage.update()
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(dimage.get_data()[0, sagittal, :, :], cmap='gray')
    plt.imshow(dlabel.get_data()[0, sagittal, :, :], cmap='tab20', alpha=0.3)
    plt.subplot(1, 3, 2)
    plt.imshow(dimage.get_data()[0, :, coronal, :], cmap='gray')
    plt.imshow(dlabel.get_data()[0, :, coronal, :], cmap='tab20', alpha=0.3)
    plt.subplot(1, 3, 3)
    plt.imshow(dimage.get_data()[0, :, :, axial], cmap='gray')
    plt.imshow(dlabel.get_data()[0, :, :, axial], cmap='tab20', alpha=0.3)
    dimage.cleanup()

    dimage.update()
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(dimage.get_data()[0, sagittal, :, :], cmap='gray')
    plt.imshow(dlabel.get_data()[0, sagittal, :, :], cmap='tab20', alpha=0.3)
    plt.subplot(1, 3, 2)
    plt.imshow(dimage.get_data()[0, :, coronal, :], cmap='gray')
    plt.imshow(dlabel.get_data()[0, :, coronal, :], cmap='tab20', alpha=0.3)
    plt.subplot(1, 3, 3)
    plt.imshow(dimage.get_data()[0, :, :, axial], cmap='gray')
    plt.imshow(dlabel.get_data()[0, :, :, axial], cmap='tab20', alpha=0.3)
    dimage.cleanup()

    plt.show()
