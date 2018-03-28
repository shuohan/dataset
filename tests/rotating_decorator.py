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
from network_utils.transformers import Rotator


filepath = 'data/AT1000_image.nii.gz'
data = Data3d(filepath, get_data_on_the_fly=False)
filepath = 'data/AT1000_mask.nii.gz'
mask = Data3d(filepath, get_data_on_the_fly=False)
filepath = 'data/AT1000_label.nii.gz'
label = Data3d(filepath, get_data_on_the_fly=False)

rotation_point = np.array(center_of_mass(mask.get_data()))
data_rotator = Rotator(max_angle=20, point=rotation_point, order=1)
mask_rotator = Rotator(max_angle=20, point=rotation_point, order=0)
label_rotator = Rotator(max_angle=20, point=rotation_point, order=0)
data_rotator.share(label_rotator, mask_rotator)

print(data_rotator._x_angle, mask_rotator._x_angle, data_rotator.order,
      mask_rotator.order)
data_rotator.update()
print(data_rotator._x_angle, mask_rotator._x_angle, data_rotator.order,
      mask_rotator.order)

data = Transforming3d(data, data_rotator, get_data_on_the_fly=True)
mask = Transforming3d(mask, mask_rotator, get_data_on_the_fly=True)
data = Cropping3d(data, mask, (128, 96, 96), get_data_on_the_fly=True)

label = Transforming3d(label, label_rotator, get_data_on_the_fly=True)
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
rotated_data = data.get_data() 
rotated_label = label.get_data()
print('get data')
shape = rotated_data.shape
plt.subplot(2, 3, 1)
plt.imshow(rotated_data[shape[0]//2, :, :], cmap='gray')
plt.subplot(2, 3, 2)
plt.imshow(rotated_data[:, shape[1]//2, :], cmap='gray')
plt.subplot(2, 3, 3)
plt.imshow(rotated_data[:, :, shape[2]//2], cmap='gray')
plt.subplot(2, 3, 4)
plt.imshow(rotated_label[shape[0]//2, :, :])
plt.subplot(2, 3, 5)
plt.imshow(rotated_label[:, shape[1]//2, :])
plt.subplot(2, 3, 6)
plt.imshow(rotated_label[:, :, shape[2]//2])
print('show')

plt.figure()
data_rotator.update()
rotated_data = data.get_data() 
rotated_label = label.get_data()
print(np.unique(rotated_label))
print('get data')
shape = rotated_data.shape
plt.subplot(2, 3, 1)
plt.imshow(rotated_data[shape[0]//2, :, :], cmap='gray')
plt.subplot(2, 3, 2)
plt.imshow(rotated_data[:, shape[1]//2, :], cmap='gray')
plt.subplot(2, 3, 3)
plt.imshow(rotated_data[:, :, shape[2]//2], cmap='gray')
plt.subplot(2, 3, 4)
plt.imshow(rotated_label[shape[0]//2, :, :])
plt.subplot(2, 3, 5)
plt.imshow(rotated_label[:, shape[1]//2, :])
plt.subplot(2, 3, 6)
plt.imshow(rotated_label[:, :, shape[2]//2])
plt.show()
