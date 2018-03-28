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
from network_utils.data_decorators import Deformer, Flipper

filepath = 'data/AT1000_image.nii.gz'
data1 = Data3d(filepath, get_data_on_the_fly=False)
filepath = 'data/AT1000_mask.nii.gz'
mask1 = Data3d(filepath, get_data_on_the_fly=False)
filepath = 'data/AT1000_label.nii.gz'
label1 = Data3d(filepath, get_data_on_the_fly=False)

label_pairs = [[33, 36], [43, 46], [53, 56], [63, 66], [73, 76], [74, 77],
               [75, 78], [83, 86], [84, 87], [93, 96], [103, 106]]
data_flipper = Flipper(dim=0)
mask_flipper = Flipper(dim=0)
label_flipper = Flipper(dim=0, label_pairs=label_pairs)

shape = data1.get_data().shape
data_deformer = Deformer(shape, sigma=3, scale=5, order=1)
mask_deformer = Deformer(shape, sigma=3, scale=5, order=0)
label_deformer = Deformer(shape, sigma=3, scale=5, order=0)
data_deformer.share(label_deformer, mask_deformer)

data2 = Transforming3d(data1, data_flipper, get_data_on_the_fly=False)
data2 = Transforming3d(data2, data_deformer, get_data_on_the_fly=True)
mask2 = Transforming3d(mask1, mask_flipper, get_data_on_the_fly=False)
mask2 = Transforming3d(mask2, mask_deformer, get_data_on_the_fly=True)
data1 = Cropping3d(data1, mask1, (128, 96, 96), get_data_on_the_fly=False)
data2 = Cropping3d(data2, mask2, (128, 96, 96), get_data_on_the_fly=True)

label2 = Transforming3d(label1, label_flipper, get_data_on_the_fly=False)
label2 = Transforming3d(label2, label_deformer, get_data_on_the_fly=True)
label1 = Cropping3d(label1, mask1, (128, 96, 96), get_data_on_the_fly=False)
label2 = Cropping3d(label2, mask2, (128, 96, 96), get_data_on_the_fly=True)

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

plt.figure()
print(np.sum(data_deformer._x_deform), np.sum(label_deformer._x_deform))
transformed_data1 = data1.get_data()
transformed_data1 = transformed_data1 / np.max(transformed_data1) * 100
transformed_label1 = label1.get_data()
transformed_data2 = data2.get_data() 
transformed_data2 = transformed_data2 / np.max(transformed_data2) * 100
transformed_label2 = label2.get_data()
print('get data')
shape = transformed_data1.shape
plt.subplot(3, 3, 1)
image = np.hstack([transformed_data1[shape[0]//2, :, :],
                   transformed_label1[shape[0]//2, :, :]])
plt.imshow(image, cmap='gray')
plt.subplot(3, 3, 2)
image = np.hstack([transformed_data1[:, shape[1]//2, :],
                   transformed_label1[:, shape[1]//2, :]])
plt.imshow(image, cmap='gray')
plt.subplot(3, 3, 3)
image = np.hstack([transformed_data1[:, :, shape[2]//2],
                   transformed_label1[:, :, shape[2]//2]])
plt.imshow(image, cmap='gray')

plt.subplot(3, 3, 4)
image = np.hstack([transformed_data2[shape[0]//2, :, :],
                   transformed_label2[shape[0]//2, :, :]])
plt.imshow(image, cmap='gray')
plt.subplot(3, 3, 5)
image = np.hstack([transformed_data2[:, shape[1]//2, :],
                   transformed_label2[:, shape[1]//2, :]])
plt.imshow(image, cmap='gray')
plt.subplot(3, 3, 6)
image = np.hstack([transformed_data2[:, :, shape[2]//2],
                   transformed_label2[:, :, shape[2]//2]])
plt.imshow(image, cmap='gray')

data2.update()
print(np.sum(data_deformer._x_deform), np.sum(label_deformer._x_deform))
transformed_data2 = data2.get_data()
transformed_data2 = transformed_data2 / np.max(transformed_data2) * 100
transformed_label2 = label2.get_data()
plt.subplot(3, 3, 7)
image = np.hstack([transformed_data2[shape[0]//2, :, :],
                   transformed_label2[shape[0]//2, :, :]])
plt.imshow(image, cmap='gray')
plt.subplot(3, 3, 8)
image = np.hstack([transformed_data2[:, shape[1]//2, :],
                   transformed_label2[:, shape[1]//2, :]])
plt.imshow(image, cmap='gray')
plt.subplot(3, 3, 9)
image = np.hstack([transformed_data2[:, :, shape[2]//2],
                   transformed_label2[:, :, shape[2]//2]])
plt.imshow(image, cmap='gray')

plt.show()
