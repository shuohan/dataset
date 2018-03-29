#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt
from time import time
from scipy.ndimage.measurements import center_of_mass

from network_utils.data import Data3d
from network_utils.data_decorators import Cropping3d, Flipping3d
from network_utils.transformers import Flipper


filepath = 'data/AT1000_image.nii.gz'
data = Data3d(filepath, get_data_on_the_fly=False)
filepath = 'data/AT1000_mask.nii.gz'
mask = Data3d(filepath, get_data_on_the_fly=False)
filepath = 'data/AT1000_label.nii.gz'
label = Data3d(filepath, get_data_on_the_fly=False)

label_pairs = [[33, 36], [43, 46], [53, 56], [63, 66], [73, 76], [74, 77],
               [75, 78], [83, 86], [84, 87], [93, 96], [103, 106]]
flipper = Flipper(dim=1)

data = Flipping3d(data, flipper, get_data_on_the_fly=True)
mask = Flipping3d(mask, flipper, get_data_on_the_fly=True)
data = Cropping3d(data, mask, (128, 96, 96), get_data_on_the_fly=True)

label = Flipping3d(label, flipper, get_data_on_the_fly=True,
                   label_pairs=label_pairs)
label = Cropping3d(label, mask, (128, 96, 96), get_data_on_the_fly=True)

start_time = time()
data.get_data()
end_time = time()
print('first load', end_time - start_time)

start_time = time()
data.get_data()
end_time = time()
print('second load', end_time - start_time)

start_time = time()
data.get_data()
end_time = time()
print('third load', end_time - start_time)

alpha = 0.7
plt.figure()
flipped_data = data.get_data()[0, ...]
flipped_label = label.get_data()[0, ...]
print('get data')
shape = flipped_data.shape
plt.subplot(1, 3, 1)
plt.imshow(flipped_data[shape[0]//2, :, :], cmap='gray', alpha=0.7)
plt.imshow(flipped_label[shape[0]//2, :, :], alpha=1-alpha)
plt.subplot(1, 3, 2)
plt.imshow(flipped_data[:, shape[1]//2, :], cmap='gray', alpha=0.7)
plt.imshow(flipped_label[:, shape[1]//2, :], alpha=1-alpha)
plt.subplot(1, 3, 3)
plt.imshow(flipped_data[:, :, shape[2]//2], cmap='gray', alpha=0.7)
plt.imshow(flipped_label[:, :, shape[2]//2], alpha=1-alpha)
print('show')

plt.figure()
flipped_data = data.get_data()[0, ...]
flipped_label = label.get_data()[0, ...]
print(np.unique(flipped_label))
print('get data')
plt.subplot(1, 3, 1)
plt.imshow(flipped_data[shape[0]//2, :, :], cmap='gray', alpha=0.7)
plt.imshow(flipped_label[shape[0]//2, :, :], alpha=1-alpha)
plt.subplot(1, 3, 2)
plt.imshow(flipped_data[:, shape[1]//2, :], cmap='gray', alpha=0.7)
plt.imshow(flipped_label[:, shape[1]//2, :], alpha=1-alpha)
plt.subplot(1, 3, 3)
plt.imshow(flipped_data[:, :, shape[2]//2], cmap='gray', alpha=0.7)
plt.imshow(flipped_label[:, :, shape[2]//2], alpha=1-alpha)
plt.show()
