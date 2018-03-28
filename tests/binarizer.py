#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt
from time import time

from network_utils.data import Data3d
from network_utils.data_decorators import Binarizing3d, Transforming3d
from network_utils.data_decorators import Cropping3d
from network_utils.label_image_binarizer import LabelImageBinarizer
from network_utils.transformers import Deformer


filepath = 'data/AT1000_image.nii.gz'
data = Data3d(filepath, get_data_on_the_fly=False)
filepath = 'data/AT1000_mask.nii.gz'
mask = Data3d(filepath, get_data_on_the_fly=False)
filepath = 'data/AT1000_label.nii.gz'
label = Data3d(filepath, get_data_on_the_fly=False)

shape = data.get_data().shape
data_deformer = Deformer(shape, sigma=3, scale=20, order=1)
mask_deformer = Deformer(shape, sigma=3, scale=20, order=0)
label_deformer = Deformer(shape, sigma=3, scale=20, order=0)
data_deformer.share(label_deformer, mask_deformer)

data = Transforming3d(data, data_deformer, get_data_on_the_fly=True)
mask = Transforming3d(mask, mask_deformer, get_data_on_the_fly=True)
data = Cropping3d(data, mask, (128, 96, 96), get_data_on_the_fly=True)

binarizer = LabelImageBinarizer()
label = Transforming3d(label, label_deformer, get_data_on_the_fly=True)
label = Cropping3d(label, mask, (128, 96, 96), get_data_on_the_fly=True)
label = Binarizing3d(label, binarizer, get_data_on_the_fly=True)

start_time = time()
label.get_data()
end_time = time()
print('first load', end_time - start_time)

start_time = time()
label.get_data()
end_time = time()
print('second load', end_time - start_time)

start_time = time()
label.get_data()
end_time = time()
print('third load', end_time - start_time)

plt.figure()
deformed_data = data.get_data() 
deformed_label = label.get_data()
print('get data')
shape = deformed_data.shape
plt.subplot(2, 3, 1)
plt.imshow(deformed_data[shape[0]//2, :, :], cmap='gray')
plt.subplot(2, 3, 2)
plt.imshow(deformed_data[:, shape[1]//2, :], cmap='gray')
plt.subplot(2, 3, 3)
plt.imshow(deformed_data[:, :, shape[2]//2], cmap='gray')
plt.subplot(2, 3, 4)
plt.imshow(deformed_label[1, shape[0]//2, :, :])
plt.subplot(2, 3, 5)
plt.imshow(deformed_label[1, :, shape[1]//2, :])
plt.subplot(2, 3, 6)
plt.imshow(deformed_label[1, :, :, shape[2]//2])
print('show')

plt.figure()
data_deformer.update()
deformed_data = data.get_data() 
deformed_label = label.get_data()
print(np.unique(deformed_label))
print('get data')
shape = deformed_data.shape
plt.subplot(2, 3, 1)
plt.imshow(deformed_data[shape[0]//2, :, :], cmap='gray')
plt.subplot(2, 3, 2)
plt.imshow(deformed_data[:, shape[1]//2, :], cmap='gray')
plt.subplot(2, 3, 3)
plt.imshow(deformed_data[:, :, shape[2]//2], cmap='gray')
plt.subplot(2, 3, 4)
plt.imshow(deformed_label[1, shape[0]//2, :, :])
plt.subplot(2, 3, 5)
plt.imshow(deformed_label[1, :, shape[1]//2, :])
plt.subplot(2, 3, 6)
plt.imshow(deformed_label[1, :, :, shape[2]//2])
plt.show()
