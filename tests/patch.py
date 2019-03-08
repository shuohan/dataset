#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

from dataset import ImageLoader, Dataset, RandomPipeline, Config


dirname = 'data'
image_ind = 5

# crop

loader = ImageLoader(dirname, id='tmc')
loader.load('image')
loader.load('label')
loader.load('mask')
dataset = Dataset(images=loader.images)

pipeline = RandomPipeline()
pipeline.register('cropping')
pipeline.register('patch')
dataset.add_pipeline(pipeline)

image, label = dataset[image_ind]

def show(image, label):
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

show(image, label)

# no crop

dataset = Dataset(images=loader.images)
pipeline = RandomPipeline()
pipeline.register('resizing')
pipeline.register('patch')
dataset.add_pipeline(pipeline)
image, label, mask = dataset[image_ind]
show(image, label)

plt.show()
