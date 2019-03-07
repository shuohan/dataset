#!/usr/bin/env python
# -*- coding: utf-8 -*-

from network_utils.images import ImageLoader, ImageType
from network_utils.datasets import Dataset
from network_utils.pipelines import RandomPipeline
from network_utils.configs import Config


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
