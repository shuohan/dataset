#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from scipy.ndimage.measurements import center_of_mass
from memory_profiler import profile
import numpy as np

from network_utils.data import Image3d, Label3d, Interpolating3d, Transforming3d
from network_utils.transformers import Scaler, Cropper


image_path = 'data/at1000_image.nii.gz'
label_path = 'data/at1000_label.nii.gz'
mask_path = 'data/at1000_mask.nii.gz'

@profile
def test(max_scale=2, cropping_shape=(128, 96, 96), on_the_fly=True,
         around_mask=True):
    """Test Rotater"""
    print('Maximum scale:', max_scale)
    print('On the fly:', on_the_fly)
    image = Image3d(image_path, on_the_fly=on_the_fly)
    label = Label3d(label_path, on_the_fly=on_the_fly)
    mask = Label3d(mask_path, on_the_fly=on_the_fly)
    if around_mask:
        point = np.array(center_of_mass(np.squeeze(mask.get_data())))
    else:
        point = None
    print('Scaling center:', point)

    scaler = Scaler(max_scale=max_scale, point=point)
    simage = Interpolating3d(image, scaler, on_the_fly=on_the_fly)
    slabel = Interpolating3d(label, scaler, on_the_fly=on_the_fly)
    smask = Interpolating3d(mask, scaler, on_the_fly=on_the_fly)

    cropper = Cropper(smask, cropping_shape)
    cimage = Transforming3d(simage, cropper, on_the_fly=on_the_fly)
    clabel = Transforming3d(slabel, cropper, on_the_fly=on_the_fly)

    return cimage, clabel

if __name__ == "__main__":
    simage, slabel = test(on_the_fly=True, around_mask=True)

    sagittal = 48
    axial = 48
    coronal = 64

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

    simage, slabel = test(on_the_fly=True, around_mask=False)

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
