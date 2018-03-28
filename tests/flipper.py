#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt
from time import time
from scipy.ndimage.measurements import center_of_mass

from network_utils.data import Data3d
from network_utils.data_decorators import Cropping3d, Transforming3d
from network_utils.data_decorators import Flipper

filepath = 'data/AT1000_image.nii.gz'
data = Data3d(filepath, get_data_on_the_fly=False)
filepath = 'data/AT1000_mask.nii.gz'
mask = Data3d(filepath, get_data_on_the_fly=False)
filepath = 'data/AT1000_label.nii.gz'
label = Data3d(filepath, get_data_on_the_fly=False)

label_pairs = [[33, 36], [43, 46], [53, 56], [63, 66], [73, 76], [74, 77],
               [75, 78], [83, 86], [84, 87], [93, 96], [103, 106]]
data_flipper = Flipper(dim=0)
mask_flipper = Flipper(dim=0)
label_flipper = Flipper(dim=0, label_pairs=label_pairs)

data = Transforming3d(data, data_flipper, get_data_on_the_fly=True)
mask = Transforming3d(mask, mask_flipper, get_data_on_the_fly=True)
data = Cropping3d(data, mask, (128, 96, 96), get_data_on_the_fly=True)

label = Transforming3d(label, label_flipper, get_data_on_the_fly=True)
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

plt.figure()
flipped_data = data.get_data() 
flipped_label = label.get_data()
print('get data')
shape = flipped_data.shape
plt.subplot(2, 3, 1)
plt.imshow(flipped_data[shape[0]//2, :, :], cmap='gray')
plt.subplot(2, 3, 2)
plt.imshow(flipped_data[:, shape[1]//2, :], cmap='gray')
plt.subplot(2, 3, 3)
plt.imshow(flipped_data[:, :, shape[2]//2], cmap='gray')
plt.subplot(2, 3, 4)
plt.imshow(flipped_label[shape[0]//2, :, :])
plt.subplot(2, 3, 5)
plt.imshow(flipped_label[:, shape[1]//2, :])
plt.subplot(2, 3, 6)
plt.imshow(flipped_label[:, :, shape[2]//2])
print('show')

plt.figure()
data_flipper.update()
flipped_data = data.get_data() 
flipped_label = label.get_data()
print(np.unique(flipped_label))
print('get data')
shape = flipped_data.shape
plt.subplot(2, 3, 1)
plt.imshow(flipped_data[shape[0]//2, :, :], cmap='gray')
plt.subplot(2, 3, 2)
plt.imshow(flipped_data[:, shape[1]//2, :], cmap='gray')
plt.subplot(2, 3, 3)
plt.imshow(flipped_data[:, :, shape[2]//2], cmap='gray')
plt.subplot(2, 3, 4)
plt.imshow(flipped_label[shape[0]//2, :, :])
plt.subplot(2, 3, 5)
plt.imshow(flipped_label[:, shape[1]//2, :])
plt.subplot(2, 3, 6)
plt.imshow(flipped_label[:, :, shape[2]//2])
plt.show()
