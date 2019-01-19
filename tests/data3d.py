#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from time import time
from memory_profiler import profile

from network_utils.data import Data3d

filepath = 'data/at1000_image.nii.gz'

@profile
def test(on_the_fly):
    """Teset Data3d"""
    print('On the fly:', on_the_fly)
    data = Data3d(filepath, on_the_fly=on_the_fly)

    start_time = time()
    data.get_data()
    end_time = time()
    print('First load', end_time - start_time)

    start_time = time()
    data.get_data()
    end_time = time()
    print('Second load', end_time - start_time)

    return data

if __name__ == "__main__":
    test(True) 
    data = test(False)

    plt.figure()
    shape = data.get_data().shape[1:]
    plt.subplot(1, 3, 1)
    plt.imshow(data.get_data()[0, shape[0]//2, :, :], cmap='gray')
    plt.subplot(1, 3, 2)
    plt.imshow(data.get_data()[0, :, shape[1]//2, :], cmap='gray')
    plt.subplot(1, 3, 3)
    plt.imshow(data.get_data()[0, :, :, shape[2]//2], cmap='gray')
    plt.show()
