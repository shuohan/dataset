#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from time import time
from scipy.ndimage.measurements import center_of_mass
from memory_profiler import profile

from network_utils.data import Data3d, Transforming3d
from network_utils.transformers import Deformer


image_path = 'data/at1000_image.nii.gz'
label_path = 'data/at1000_label.nii.gz'

@profile
def test(scale=30, sigma=5, on_the_fly=True):
    """Test Deformer"""
    print('Scale:', scale)
    print('Sigma:', sigma)
    print('On the fly:', on_the_fly)
    image = Data3d(image_path, on_the_fly=on_the_fly)
    label = Data3d(label_path, on_the_fly=on_the_fly)
    shape = image.get_data().shape
    deformer = Deformer(shape, sigma, scale)
    deformer.update()

    dimage = Transforming3d(image, deformer, on_the_fly=on_the_fly)
    dlabel = Transforming3d(label, deformer, on_the_fly=on_the_fly)

    start_time = time()
    dimage.get_data()
    dlabel.get_data()
    end_time = time()
    print('First rotate', end_time - start_time)

    start_time = time()
    dimage.get_data()
    dlabel.get_data()
    end_time = time()
    print('Second rotate', end_time - start_time)

    return dimage, dlabel

if __name__ == "__main__":
    test(on_the_fly=True)
    dimage, dlabel = test(on_the_fly=False)

    sagittal = 90
    axial = 50
    coronal = 150
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
    plt.show()
