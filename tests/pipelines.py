#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.morphology import binary_dilation

from dataset.images import ImageLoader, ImageType
from dataset.datasets import Dataset
from dataset.pipelines import RandomPipeline
from dataset.configs import Config

dirname = 'data'
image_ind = 5

print('no cropping')
loader = ImageLoader(dirname, id='tmc')
loader.load(ImageType.image, ImageType.label, ImageType.mask)
dataset = Dataset(images=loader.images, verbose=Config().verbose)

pipeline = RandomPipeline()
dataset.add_pipeline(pipeline)

Config().image_shape = dataset[0][0].shape

image, label, mask = dataset[image_ind]
print(image.dtype, label.dtype, mask.dtype)

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

# ------------------------------------------------------------------------------ 

print('bounding box')
loader = ImageLoader(dirname, id='tmc')
loader.load(ImageType.image, ImageType.label)
loader.load(ImageType.bounding_box, ImageType.mask)
dataset = Dataset(loader.images, verbose=Config().verbose)

pipeline = RandomPipeline()
pipeline.register('scaling')
pipeline.register('rotation')
pipeline.register('deformation')
pipeline.register('cropping')
dataset.add_pipeline(pipeline)

image, label, bbox = dataset[image_ind]
print(image.dtype, label.dtype, bbox.dtype)
shape = image.shape
mask = np.zeros(image.shape, dtype=bool)
mask[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]] = 1
mask = binary_dilation(mask) ^ mask
plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(image[shape[0]//2, :, :], cmap='gray')
plt.imshow(label[shape[0]//2, :, :], alpha=0.3)
plt.imshow(mask[shape[0]//2, :, :], alpha=0.3, cmap='autumn')
plt.subplot(1, 3, 2)
plt.imshow(image[:, shape[1]//2, :], cmap='gray')
plt.imshow(label[:, shape[1]//2, :], alpha=0.3)
plt.imshow(mask[:, shape[1]//2, :], alpha=0.3, cmap='autumn')
plt.subplot(1, 3, 3)
plt.imshow(image[:, :, shape[2]//2], cmap='gray')
plt.imshow(label[:, :, shape[2]//2], alpha=0.3)
plt.imshow(mask[:, :, shape[2]//2], alpha=0.3, cmap='autumn')
plt.title('bounding box')

# ------------------------------------------------------------------------------ 

print('flipping')
loader = ImageLoader(dirname, id='tmc')
loader.load(ImageType.image, ImageType.label, ImageType.mask)
dataset = Dataset(loader.images, verbose=Config().verbose)

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

augmentation = ['rotation', 'scaling', 'translation', 'deformation']
for aug in augmentation:
    print(aug)
    loader = ImageLoader(dirname, id='tmc')
    loader.load(ImageType.image, ImageType.label, ImageType.mask)
    dataset = Dataset(images=loader.images, verbose=Config().verbose)

    pipeline = RandomPipeline()
    pipeline.register(aug)
    if aug != 'translation':
        pipeline.register('cropping')
    pipeline.register('label_normalization')
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
