#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from dataset import ImageLoader, Dataset, RandomPipeline, Config


dirname = 'data'
image_ind = 5

loader = ImageLoader(dirname, id='tmc')
loader.load('image')
loader.load('label')
loader.load('mask')
dataset = Dataset(images=loader.images)

pipeline = RandomPipeline()
pipeline.register('sigmoid_intensity', 'cropping')
dataset.add_pipeline(pipeline)

image, label = dataset[image_ind]
print(np.unique(label))
shape = image.shape[1:]

dataset = Dataset(images=loader.images)
pipeline = RandomPipeline()
pipeline.register('cropping')
dataset.add_pipeline(pipeline)
orig_image, label = dataset[image_ind]
print(np.unique(label))

plt.figure()
plt.subplot(2, 3, 1)
plt.imshow(image[0, shape[0]//2, :, :], cmap='gray')
plt.subplot(2, 3, 2)
plt.imshow(image[0, :, shape[1]//2, :], cmap='gray')
plt.subplot(2, 3, 3)
plt.imshow(image[0, :, :, shape[2]//2], cmap='gray')

plt.subplot(2, 3, 4)
plt.imshow(orig_image[0, shape[0]//2, :, :], cmap='gray')
plt.subplot(2, 3, 5)
plt.imshow(orig_image[0, :, shape[1]//2, :], cmap='gray')
plt.subplot(2, 3, 6)
plt.imshow(orig_image[0, :, :, shape[2]//2], cmap='gray')
plt.show()
