#!/usr/bin/env python
# -*- coding: utf-8 -*-

from network_utils.datasets import ImageDataset, Delineated, Masked


dirname = 'data'
dataset = ImageDataset()
dataset = Delineated(dataset)
dataset = Masked(dataset)

dataset.add_images(dirname, id='tmc')
print(dataset.images['tmc/at1000'][0])
print(dataset.images['tmc/at1000'][1].pairs)
print(dataset.images['tmc/at1000'][1].labels)
print(dataset)

dirname = 'data'
dataset = ImageDataset()
dataset = Masked(dataset)

dataset.add_images(dirname, id='tmc')
print(dataset.images['tmc/at1000'][0])
print(dataset)
