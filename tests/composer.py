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
from network_utils.data_decorators import Flipping3d
from network_utils.transformers import Deformer, Flipper


filepath = 'data/AT1000_image.nii.gz'
data1 = Data3d(filepath, get_data_on_the_fly=False)
filepath = 'data/AT1000_mask.nii.gz'
mask1 = Data3d(filepath, get_data_on_the_fly=False)
filepath = 'data/AT1000_label.nii.gz'
label1 = Data3d(filepath, get_data_on_the_fly=False)

label_pairs = [[33, 36], [43, 46], [53, 56], [63, 66], [73, 76], [74, 77],
               [75, 78], [83, 86], [84, 87], [93, 96], [103, 106]]
flipper = Flipper(dim=1)

shape = data1.get_data().shape[-3:]
deformer = Deformer(shape, sigma=3, scale=8)

data2 = Flipping3d(data1, flipper, get_data_on_the_fly=False)
data2 = Interpolating3d(data2, deformer, get_data_on_the_fly=True, order=1)
mask2 = Flipping3d(mask1, flipper, get_data_on_the_fly=False)
mask2 = Interpolating3d(mask2, deformer, get_data_on_the_fly=True, order=0)
data1 = Cropping3d(data1, mask1, (128, 96, 96), get_data_on_the_fly=False)
data2 = Cropping3d(data2, mask2, (128, 96, 96), get_data_on_the_fly=True)

label2 = Flipping3d(label1, flipper, get_data_on_the_fly=False,
                    label_pairs=label_pairs)
label2 = Interpolating3d(label2, deformer, get_data_on_the_fly=True, order=0)
label1 = Cropping3d(label1, mask1, (128, 96, 96), get_data_on_the_fly=False)
label2 = Cropping3d(label2, mask2, (128, 96, 96), get_data_on_the_fly=True)

data1.update()
data2.update()

start_time = time()
data2.get_data()
end_time = time()
print('first load', end_time - start_time)

start_time = time()
data2.get_data()
end_time = time()
print('second load', end_time - start_time)

start_time = time()
data2.get_data()
end_time = time()
print('third load', end_time - start_time)

alpha = 0.7
plt.figure()
transformed_data1 = data1.get_data()[0, ...]
transformed_label1 = label1.get_data()[0, ...]
transformed_data2 = data2.get_data() [0, ...]
transformed_label2 = label2.get_data()[0, ...]
print('get data')
shape = transformed_data1.shape
plt.subplot(2, 3, 1)
image = transformed_data1[shape[0]//2, :, :]
label = transformed_label1[shape[0]//2, :, :]
# plt.imshow(image, cmap='gray', alpha=alpha)
plt.imshow(label, alpha=1-alpha)
plt.subplot(2, 3, 2)
image = transformed_data1[:, shape[1]//2, :]
label = transformed_label1[:, shape[1]//2, :]
# plt.imshow(image, cmap='gray', alpha=alpha)
plt.imshow(label, alpha=1-alpha)
plt.subplot(2, 3, 3)
image = transformed_data1[:, :, shape[2]//2]
label = transformed_label1[:, :, shape[2]//2]
# plt.imshow(image, cmap='gray', alpha=alpha)
plt.imshow(label, alpha=1-alpha)

plt.subplot(2, 3, 4)
image = transformed_data2[shape[0]//2, :, :]
label = transformed_label2[shape[0]//2, :, :]
# plt.imshow(image, cmap='gray', alpha=alpha)
plt.imshow(label, alpha=1-alpha)
plt.subplot(2, 3, 5)
image = transformed_data2[:, shape[1]//2, :]
label = transformed_label2[:, shape[1]//2, :]
# plt.imshow(image, cmap='gray', alpha=alpha)
plt.imshow(label, alpha=1-alpha)
plt.subplot(2, 3, 6)
image = transformed_data2[:, :, shape[2]//2]
label = transformed_label2[:, :, shape[2]//2]
# plt.imshow(image, cmap='gray', alpha=alpha)
plt.imshow(label, alpha=1-alpha)

plt.show()
