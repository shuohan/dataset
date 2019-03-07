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
from dataset.configs import Config

dirname = 'data'
image_ind = 5

loader = ImageLoader(dirname, id='tmc')
loader.load('image', 'label')
dataset = Dataset(images=loader.images, verbose=Config().verbose)
pipeline = RandomPipeline()
dataset.add_pipeline(pipeline)
data_loader = DataLoader(dataset)
for data in data_loader:
    print(type(data[0]))
    print(data[1].shape)
    break
