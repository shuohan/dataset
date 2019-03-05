#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.morphology import binary_dilation

from network_utils.datasets import Dataset, Delineated, Masked, Located
from network_utils.pipelines import RandomPipeline
from network_utils.configs import Config

dirname = 'data'
image_ind = 5

dataset = Dataset(verbose=True)
dataset = Delineated(dataset)
dataset = Located(dataset)
dataset = Masked(dataset)
dataset.add_images(dirname, id='tmc')
print(dataset)
dataset1, dataset2 = dataset.split([1, 3, 5, 6])

pipeline1 = RandomPipeline()
pipeline1.register('flipping')
pipeline1.register('rotation')
pipeline1.register('cropping')
dataset1.add_pipeline(pipeline1)

pipeline2 = RandomPipeline()
pipeline2.register('cropping')
dataset2.add_pipeline(pipeline2)

print('*' * 80)
print(dataset1)
print('*' * 80)
print(dataset2)
print(len(dataset1), len(dataset2))

image1, label1, bbox1 = dataset1[2]
image2, label2, bbox2 = dataset2[2]
print(bbox1, bbox2)
shape = image1.shape

plt.figure()
plt.subplot(2, 3, 1)
plt.imshow(image1[shape[0]//2, :, :], cmap='gray')
plt.imshow(label1[shape[0]//2, :, :], alpha=0.3)
plt.subplot(2, 3, 2)
plt.imshow(image1[:, shape[1]//2, :], cmap='gray')
plt.imshow(label1[:, shape[1]//2, :], alpha=0.3)
plt.subplot(2, 3, 3)
plt.imshow(image1[:, :, shape[2]//2], cmap='gray')
plt.imshow(label1[:, :, shape[2]//2], alpha=0.3)

plt.subplot(2, 3, 4)
plt.imshow(image2[shape[0]//2, :, :], cmap='gray')
plt.imshow(label2[shape[0]//2, :, :], alpha=0.3)
plt.subplot(2, 3, 5)
plt.imshow(image2[:, shape[1]//2, :], cmap='gray')
plt.imshow(label2[:, shape[1]//2, :], alpha=0.3)
plt.subplot(2, 3, 6)
plt.imshow(image2[:, :, shape[2]//2], cmap='gray')
plt.imshow(label2[:, :, shape[2]//2], alpha=0.3)

plt.show()
