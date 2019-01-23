#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from time import time
from memory_profiler import profile

from network_utils.data import Image3d, Label3d, Transforming3d
from network_utils.transformers import Translater


image_path = 'data/at1000_image.nii.gz'
label_path = 'data/at1000_label.nii.gz'

@profile
def test(max_translation=50, on_the_fly=True):
    """Test Translater"""
    print('Maximum translation:', max_translation)
    print('On the fly:', on_the_fly)
    image = Image3d(image_path, on_the_fly=on_the_fly)
    label = Label3d(label_path, on_the_fly=on_the_fly)
    translater = Translater(max_trans=max_translation)
    timage = Transforming3d(image, translater, on_the_fly=on_the_fly)
    tlabel = Transforming3d(label, translater, on_the_fly=on_the_fly)

    translater.update()
    start_time = time()
    timage.get_data()
    tlabel.get_data()
    end_time = time()
    print('First translation', end_time - start_time)
    translater.cleanup()

    translater.update()
    start_time = time()
    timage.get_data()
    tlabel.get_data()
    end_time = time()
    print('Second translate', end_time - start_time)
    translater.cleanup()

    return timage, tlabel

if __name__ == "__main__":
    timage, tlabel = test(on_the_fly=True)
    test(on_the_fly=False)

    sagittal = 90
    axial = 50
    coronal = 150

    timage.update()
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(timage.get_data()[0, sagittal, :, :], cmap='gray')
    plt.imshow(tlabel.get_data()[0, sagittal, :, :], cmap='tab20', alpha=0.3)
    plt.subplot(1, 3, 2)
    plt.imshow(timage.get_data()[0, :, coronal, :], cmap='gray')
    plt.imshow(tlabel.get_data()[0, :, coronal, :], cmap='tab20', alpha=0.3)
    plt.subplot(1, 3, 3)
    plt.imshow(timage.get_data()[0, :, :, axial], cmap='gray')
    plt.imshow(tlabel.get_data()[0, :, :, axial], cmap='tab20', alpha=0.3)
    timage.cleanup()

    timage.update()
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(timage.get_data()[0, sagittal, :, :], cmap='gray')
    plt.imshow(tlabel.get_data()[0, sagittal, :, :], cmap='tab20', alpha=0.3)
    plt.subplot(1, 3, 2)
    plt.imshow(timage.get_data()[0, :, coronal, :], cmap='gray')
    plt.imshow(tlabel.get_data()[0, :, coronal, :], cmap='tab20', alpha=0.3)
    plt.subplot(1, 3, 3)
    plt.imshow(timage.get_data()[0, :, :, axial], cmap='gray')
    plt.imshow(tlabel.get_data()[0, :, :, axial], cmap='tab20', alpha=0.3)
    timage.cleanup()
    plt.show()
