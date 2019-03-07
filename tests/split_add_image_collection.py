#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataset.images import ImageLoader, ImageType
from dataset.datasets import Dataset
from dataset.pipelines import RandomPipeline
from dataset.configs import Config


dirname = 'data'
loader = ImageLoader(dirname, id='tmc')
loader.load(ImageType.image, ImageType.label)
loader.load(ImageType.bounding_box, ImageType.mask)
images1, images2 = loader.images.split([0, 3, 5, 8])
dataset1 = Dataset(images1, verbose=Config().verbose)
dataset2 = Dataset(images2, verbose=Config().verbose)

print(len(dataset1))
print(dataset1)
print()
print(len(dataset2))
print(dataset2)
print()

images = images1 + images2
dataset = Dataset(images, verbose=Config().verbose)

print(len(dataset))
print(dataset)
