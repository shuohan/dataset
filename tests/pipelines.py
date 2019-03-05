#!/usr/bin/env python
# -*- coding: utf-8 -*-

from network_utils.pipelines import RandomPipeline
from network_utils.datasets import Dataset, Delineated, Masked
import matplotlib.pyplot as plt

dirname = 'data'
dataset = Dataset(verbose=True)
dataset = Delineated(dataset)
dataset = Masked(dataset)
dataset.add_images(dirname, id='tmc')

from network_utils.workers import WorkerTypeMapping

worker_types = WorkerTypeMapping()

pipeline = RandomPipeline()
pipeline.register('flipping')
pipeline.register('rotation')
pipeline.register('scaling')
pipeline.register('cropping')
dataset.add_pipeline(pipeline)
print('Length of dataset:', len(dataset))
print(dataset)

dirname = 'data'
dataset2 = Dataset(verbose=True)
dataset2 = Delineated(dataset2)
dataset2 = Masked(dataset2)
dataset2.add_images(dirname, id='tmc')

pipeline2 = RandomPipeline()
pipeline2.register('rotation')
pipeline2.register('scaling')
pipeline2.register('cropping')
dataset2.add_pipeline(pipeline2)

image, label = dataset[5]
shape = image.shape
plt.subplot(2, 3, 1)
plt.imshow(image[shape[0]//2, :, :], cmap='gray')
plt.imshow(label[shape[0]//2, :, :], alpha=0.3)
plt.subplot(2, 3, 2)
plt.imshow(image[:, shape[1]//2, :], cmap='gray')
plt.imshow(label[:, shape[1]//2, :], alpha=0.3)
plt.subplot(2, 3, 3)
plt.imshow(image[:, :, shape[2]//2], cmap='gray')
plt.imshow(label[:, :, shape[2]//2], alpha=0.3)

image, label = dataset2[5]
shape = image.shape
plt.subplot(2, 3, 4)
plt.imshow(image[shape[0]//2, :, :], cmap='gray')
plt.imshow(label[shape[0]//2, :, :], alpha=0.3)
plt.subplot(2, 3, 5)
plt.imshow(image[:, shape[1]//2, :], cmap='gray')
plt.imshow(label[:, shape[1]//2, :], alpha=0.3)
plt.subplot(2, 3, 6)
plt.imshow(image[:, :, shape[2]//2], cmap='gray')
plt.imshow(label[:, :, shape[2]//2], alpha=0.3)

plt.show()
