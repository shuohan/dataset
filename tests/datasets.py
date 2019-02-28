#!/usr/bin/env python
# -*- coding: utf-8 -*-

from network_utils.datasets import ImageDataset


dirname = 'data'
dataset = ImageDataset()

dataset.add_images(dirname, id='tmc')
print(dataset.images['tmc/at1000'][0])
print(dataset)
