#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt
from time import time
from scipy.ndimage.measurements import center_of_mass

from network_utils.data import Data3d
from network_utils.data_decorators import Cropping3d, Interpolating3d
from network_utils.transformers import Deformer


filepath = 'data/AT1000_image.nii.gz'
data = Data3d(filepath, get_data_on_the_fly=False)
filepath = 'data/AT1000_mask.nii.gz'
mask = Data3d(filepath, get_data_on_the_fly=False)
filepath = 'data/AT1000_label.nii.gz'
label = Data3d(filepath, get_data_on_the_fly=False)

shape = data.get_data().shape[-3:]
deformer = Deformer(shape, sigma=3, scale=20)

data = Interpolating3d(data, deformer, get_data_on_the_fly=True, order=1)
mask = Interpolating3d(mask, deformer, get_data_on_the_fly=True, order=0)
data = Cropping3d(data, mask, (128, 96, 96), get_data_on_the_fly=True)

label = Interpolating3d(label, deformer, get_data_on_the_fly=True, order=0)
label = Cropping3d(label, mask, (128, 96, 96), get_data_on_the_fly=True)

start_time = time()
deformer.update()
data.get_data()
end_time = time()
print('first load', end_time - start_time)

start_time = time()
deformer.update()
data.get_data()
end_time = time()
print('second load', end_time - start_time)

start_time = time()
deformer.update()
data.get_data()
end_time = time()
print('third load', end_time - start_time)

alpha = 0.7
plt.figure()
deformer.update()
deformed_data = data.get_data()[0, ...]
deformed_label = label.get_data()[0, ...]
shape = deformed_data.shape
plt.subplot(2, 3, 1)
plt.imshow(deformed_data[shape[0]//2, :, :], cmap='gray', alpha=alpha)
plt.imshow(deformed_label[shape[0]//2, :, :], alpha=1-alpha)
plt.subplot(2, 3, 2)
plt.imshow(deformed_data[:, shape[1]//2, :], cmap='gray', alpha=alpha)
plt.imshow(deformed_label[:, shape[1]//2, :], alpha=1-alpha)
plt.subplot(2, 3, 3)
plt.imshow(deformed_data[:, :, shape[2]//2], cmap='gray', alpha=alpha)
plt.imshow(deformed_label[:, :, shape[2]//2], alpha=1-alpha)

deformer.update()
deformed_data = data.get_data()[0, ...]
deformed_label = label.get_data()[0, ...]
print(np.unique(deformed_label))
shape = deformed_data.shape
plt.subplot(2, 3, 4)
plt.imshow(deformed_data[shape[0]//2, :, :], cmap='gray', alpha=alpha)
plt.imshow(deformed_label[shape[0]//2, :, :], alpha=1-alpha)
plt.subplot(2, 3, 5)
plt.imshow(deformed_data[:, shape[1]//2, :], cmap='gray', alpha=alpha)
plt.imshow(deformed_label[:, shape[1]//2, :], alpha=1-alpha)
plt.subplot(2, 3, 6)
plt.imshow(deformed_data[:, :, shape[2]//2], cmap='gray', alpha=alpha)
plt.imshow(deformed_label[:, :, shape[2]//2], alpha=1-alpha)
plt.show()
