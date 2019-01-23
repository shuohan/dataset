#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from time import time
from memory_profiler import profile

from network_utils.data import Image3d, Label3d, Flipping3d
from network_utils.transformers import Flipper


image_path = 'data/at1000_image.nii.gz'
label_path = 'data/at1000_label.nii.gz'
label_pairs = [[33, 36], [43, 46], [53, 56], [63, 66], [73, 76], [74, 77],
               [75, 78], [83, 86], [84, 87], [93, 96], [103, 106]]

@profile
def test(on_the_fly=True):
    """Test Flipper"""
    print('On the fly:', on_the_fly)
    image = Image3d(image_path, on_the_fly=on_the_fly)
    label = Label3d(label_path, on_the_fly=on_the_fly)
    label.value_pairs = label_pairs
    flipper = Flipper(dim=1)
    fimage = Flipping3d(image, flipper, on_the_fly=on_the_fly)
    flabel = Flipping3d(label, flipper, on_the_fly=on_the_fly)

    flipper.update()
    start_time = time()
    fimage.get_data()
    flabel.get_data()
    end_time = time()
    print('First flipper', end_time - start_time)
    flipper.cleanup()

    flipper.update()
    start_time = time()
    fimage.get_data()
    flabel.get_data()
    end_time = time()
    print('Second flipper', end_time - start_time)
    flipper.cleanup()

    return image, label, fimage, flabel

if __name__ == "__main__":
    test(on_the_fly=True)
    image, label, fimage, flabel = test(on_the_fly=False)
    a_ind = 50
    c_ind = 145
    s_ind = 90

    plt.figure()
    shape = fimage.shape[1:]
    plt.subplot(2, 3, 1)
    plt.imshow(fimage.get_data()[0, s_ind, :, :], cmap='gray')
    plt.imshow(flabel.get_data()[0, s_ind, :, :], cmap='tab20', alpha=0.3)
    plt.subplot(2, 3, 2)
    plt.imshow(fimage.get_data()[0, :, c_ind, :], cmap='gray')
    plt.imshow(flabel.get_data()[0, :, c_ind, :], cmap='tab20', alpha=0.3)
    plt.subplot(2, 3, 3)
    plt.imshow(fimage.get_data()[0, :, :, a_ind], cmap='gray')
    plt.imshow(flabel.get_data()[0, :, :, a_ind], cmap='tab20', alpha=0.3)
    plt.subplot(2, 3, 4)
    plt.imshow(image.get_data()[0, s_ind, :, :], cmap='gray')
    plt.imshow(label.get_data()[0, s_ind, :, :], cmap='tab20', alpha=0.3)
    plt.subplot(2, 3, 5)
    plt.imshow(image.get_data()[0, :, c_ind, :], cmap='gray')
    plt.imshow(label.get_data()[0, :, c_ind, :], cmap='tab20', alpha=0.3)
    plt.subplot(2, 3, 6)
    plt.imshow(image.get_data()[0, :, :, a_ind], cmap='gray')
    plt.imshow(label.get_data()[0, :, :, a_ind], cmap='tab20', alpha=0.3)
    plt.show()
