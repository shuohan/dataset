#!/usr/bin/env python
# -*- coding: utf-8 -*-

from network_utils.datasets import ImageDataset, Delineated


dirname = 'data'
dataset = ImageDataset()
dataset = Delineated(dataset)

dataset.add_images(dirname, id='tmc')
print(dataset.images['tmc/at1000'][0])
print(dataset.images['tmc/at1000'][1].pairs)
print(dataset.images['tmc/at1000'][1].labels)
print(dataset)
