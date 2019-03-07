#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.morphology import binary_dilation
from torch.utils.data import DataLoader

from network_utils.images import ImageLoader, ImageType
from network_utils.datasets import Dataset
from network_utils.pipelines import RandomPipeline
from network_utils.configs import Config

dirname = 'data'
image_ind = 5

loader = ImageLoader(dirname, id='tmc')
loader.load(ImageType.image, ImageType.label)
dataset = Dataset(images=loader.images, verbose=Config().verbose)
pipeline = RandomPipeline()
dataset.add_pipeline(pipeline)
data_loader = DataLoader(dataset)
for data in data_loader:
    print(type(data[0]))
    print(data[1].shape)
    break
