#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')

import matplotlib.pyplot as plt
from time import time

from network_utils.data import Data3d

filepath = 'data/AT1000_image.nii.gz'

data = Data3d(filepath, get_data_on_the_fly=True)

start_time = time()
data.get_data()
end_time = time()
print('get data on the fly, first load', end_time - start_time)

start_time = time()
data.get_data()
end_time = time()
print('get data on the fly, second load', end_time - start_time)

data = Data3d(filepath, get_data_on_the_fly=False)

start_time = time()
data.get_data()
end_time = time()
print('get data once, first load', end_time - start_time)

start_time = time()
data.get_data()
end_time = time()
print('get data once, second load', end_time - start_time)

plt.figure()
shape = data.get_data().shape
plt.subplot(1, 3, 1)
plt.imshow(data.get_data()[shape[0]//2, :, :], cmap='gray')
plt.subplot(1, 3, 2)
plt.imshow(data.get_data()[:, shape[1]//2, :], cmap='gray')
plt.subplot(1, 3, 3)
plt.imshow(data.get_data()[:, :, shape[2]//2], cmap='gray')
plt.show()
