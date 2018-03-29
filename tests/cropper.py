#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')

import matplotlib.pyplot as plt
from time import time

from network_utils.data import Data3d
from network_utils.data_decorators import Cropping3d


filepath = 'data/AT1000_image.nii.gz'
data = Data3d(filepath, get_data_on_the_fly=False)
filepath = 'data/AT1000_mask.nii.gz'
mask = Data3d(filepath, get_data_on_the_fly=False)
data = Cropping3d(data, mask, (128, 96, 96), get_data_on_the_fly=False)

start_time = time()
data.get_data()
end_time = time()
print('first crop', end_time - start_time)

start_time = time()
data.get_data()
end_time = time()
print('second load', end_time - start_time)

plt.figure()
shape = data.get_data().shape
plt.subplot(1, 3, 1)
plt.imshow(data.get_data()[shape[0]//2, :, :], cmap='gray')
plt.subplot(1, 3, 2)
plt.imshow(data.get_data()[:, shape[1]//2, :], cmap='gray')
plt.subplot(1, 3, 3)
plt.imshow(data.get_data()[:, :, shape[2]//2], cmap='gray')
plt.show()
