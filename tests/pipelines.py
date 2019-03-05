#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from network_utils.datasets import Dataset, Delineated, Masked
from network_utils.pipelines import RandomPipeline
from network_utils.configs import Config

dirname = 'data'
image_ind = 5

dataset = Dataset(verbose=True)
dataset = Delineated(dataset)
dataset.add_images(dirname, id='tmc')
pipeline = RandomPipeline()
dataset.add_pipeline(pipeline)

Config().image_shape = dataset[0][0].shape

image, label = dataset[image_ind]
shape = image.shape
plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(image[shape[0]//2, :, :], cmap='gray')
plt.imshow(label[shape[0]//2, :, :], alpha=0.3)
plt.subplot(1, 3, 2)
plt.imshow(image[:, shape[1]//2, :], cmap='gray')
plt.imshow(label[:, shape[1]//2, :], alpha=0.3)
plt.subplot(1, 3, 3)
plt.imshow(image[:, :, shape[2]//2], cmap='gray')
plt.imshow(label[:, :, shape[2]//2], alpha=0.3)
plt.title('no cropping')

dataset = Dataset(verbose=True)
dataset = Delineated(dataset)
dataset = Masked(dataset)
dataset.add_images(dirname, id='tmc')

pipeline = RandomPipeline()
pipeline.register('flipping')
pipeline.register('cropping')
dataset.add_pipeline(pipeline)

image, label = dataset[image_ind]
shape = image.shape
plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(image[shape[0]//2, :, :], cmap='gray')
plt.imshow(label[shape[0]//2, :, :], alpha=0.3)
plt.subplot(1, 3, 2)
plt.imshow(image[:, shape[1]//2, :], cmap='gray')
plt.imshow(label[:, shape[1]//2, :], alpha=0.3)
plt.subplot(1, 3, 3)
plt.imshow(image[:, :, shape[2]//2], cmap='gray')
plt.imshow(label[:, :, shape[2]//2], alpha=0.3)
plt.title('flipping')

# augmentation = ['rotation', 'scaling', 'translation', 'deformation']
augmentation = ['deformation']
for aug in augmentation:
    dataset = Dataset(verbose=True)
    dataset = Delineated(dataset)
    dataset = Masked(dataset)
    dataset.add_images(dirname, id='tmc')

    pipeline = RandomPipeline()
    pipeline.register(aug)
    if aug != 'translation':
        pipeline.register('cropping')
    dataset.add_pipeline(pipeline)

    image, label = dataset[image_ind][:2]
    shape = image.shape
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(image[shape[0]//2, :, :], cmap='gray')
    plt.imshow(label[shape[0]//2, :, :], alpha=0.3)
    plt.subplot(1, 3, 2)
    plt.imshow(image[:, shape[1]//2, :], cmap='gray')
    plt.imshow(label[:, shape[1]//2, :], alpha=0.3)
    plt.subplot(1, 3, 3)
    plt.imshow(image[:, :, shape[2]//2], cmap='gray')
    plt.imshow(label[:, :, shape[2]//2], alpha=0.3)
    plt.title(aug)

plt.show()
