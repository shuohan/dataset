#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataset.images import ImageLoader
from dataset.datasets import Dataset
from dataset.pipelines import RandomPipeline
from dataset.config import Config


dirname = 'data'
loader = ImageLoader(dirname, id='tmc')
loader.load('image', 'label', 'bounding_box', 'mask')
images1, images2 = loader.images.split([0, 3, 5, 8])
dataset1 = Dataset(images1)
dataset2 = Dataset(images2)

print(len(dataset1))
print(dataset1)
print()
print(len(dataset2))
print(dataset2)
print()

images = images1 + images2
dataset = Dataset(images)

print(len(dataset))
print(dataset)
