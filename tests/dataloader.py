#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.morphology import binary_dilation
from torch.utils.data import DataLoader

from dataset.images import ImageLoader
from dataset.datasets import Dataset
from dataset.pipelines import RandomPipeline
from dataset.config import Config

dirname = 'data'
image_ind = 5

loader = ImageLoader(dirname, id='tmc')
loader.load('image', 'label', 'mask', 'bounding_box')
dataset = Dataset(images=loader.images)
pipeline = RandomPipeline()
pipeline.register('resizing', 'scaling', 'cropping')
dataset.add_pipeline(pipeline)
data_loader = DataLoader(dataset)

print('dataset')
for d in dataset.images['tmc/at1000']:
    print(d.data.dtype)
    print(d.data.shape)

print('data loader')
for data in data_loader:
    for d in data:
        print(d.dtype)
        print(d.shape)
    break
