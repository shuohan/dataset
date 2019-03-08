#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.morphology import binary_dilation

from dataset import ImageLoader, Dataset, RandomPipeline, Config

dirname = 'data'
image_ind = 5

def show(image, label, title, mask=None):
    shape = image.shape[1:]
    fig = plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(image[0, shape[0]//2, :, :], cmap='gray')
    plt.imshow(label[0, shape[0]//2, :, :], alpha=0.3)
    if mask is not None:
        plt.imshow(mask[0, shape[0]//2, :, :], alpha=0.3, cmap='autumn')
    plt.subplot(1, 3, 2)
    plt.imshow(image[0, :, shape[1]//2, :], cmap='gray')
    plt.imshow(label[0, :, shape[1]//2, :], alpha=0.3)
    if mask is not None:
        plt.imshow(mask[0, :, shape[1]//2, :], alpha=0.3, cmap='autumn')
    plt.subplot(1, 3, 3)
    plt.imshow(image[0, :, :, shape[2]//2], cmap='gray')
    plt.imshow(label[0, :, :, shape[2]//2], alpha=0.3)
    if mask is not None:
        plt.imshow(mask[0, :, :, shape[2]//2], alpha=0.3, cmap='autumn')
    fig.suptitle(title)

Config().verbose = True

print('no cropping')
loader = ImageLoader(dirname, id='tmc')
loader.load('image', 'label', 'mask')
dataset = Dataset(images=loader.images)

pipeline = RandomPipeline()
pipeline.register('resizing')
dataset.add_pipeline(pipeline)

image, label, mask = dataset[image_ind]
print(image.shape)
print(image.dtype, label.dtype, mask.dtype)
show(image, label, 'no cropping')

# ------------------------------------------------------------------------------ 

print('bounding box')
loader = ImageLoader(dirname, id='tmc')
loader.load('image', 'label', 'bounding_box', 'mask')
dataset = Dataset(loader.images)

pipeline = RandomPipeline()
pipeline.register('resizing', 'scaling', 'rotation', 'deformation', 'cropping')
dataset.add_pipeline(pipeline)

image, label, bbox = dataset[image_ind]
print(image.shape)
print(image.dtype, label.dtype, bbox.dtype)

mask = np.zeros(image.shape, dtype=bool)
mask[..., bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]] = 1
mask = binary_dilation(mask) ^ mask

show(image, label, 'bounding box', mask=mask)

# ------------------------------------------------------------------------------ 

print('flipping')
loader = ImageLoader(dirname, id='tmc')
loader.load('image', 'label', 'mask')
dataset = Dataset(loader.images)

pipeline = RandomPipeline()
pipeline.register('flipping', 'cropping')
dataset.add_pipeline(pipeline)

image, label = dataset[image_ind]
print(image.shape)
show(image, label, 'flipping')

augmentation = ['rotation', 'scaling', 'translation', 'deformation']
for aug in augmentation:
    print(aug)
    loader = ImageLoader(dirname, id='tmc')
    loader.load('image', 'label', 'mask')
    dataset = Dataset(images=loader.images)

    pipeline = RandomPipeline()
    pipeline.register(aug)
    if aug != 'translation':
        pipeline.register('cropping')
    pipeline.register('label_normalization')
    dataset.add_pipeline(pipeline)

    image, label = dataset[image_ind][:2]
    print(image.shape)
    show(image, label, aug)

plt.show()
