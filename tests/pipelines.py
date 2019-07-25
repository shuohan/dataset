#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import numpy as np

from dataset import Config
from dataset.pipelines import RandomPipeline
from dataset.images import Image, Label, Mask
from plot import imshow


image_filepath = 'data/at1000_image.nii.gz'
label_filepath = 'data/at1000_label.nii.gz'
mask_filepath = 'data/at1000_mask.nii.gz'

image = Image(image_filepath, on_the_fly=False)
label = Label(label_filepath, on_the_fly=False)
mask = Mask(mask_filepath, on_the_fly=False)

Config().verbose = True

pipeline = RandomPipeline()
pipeline.register('resize')
results = pipeline.process(image, label, mask)
imshow((image, label, mask), results)
plt.gcf().suptitle(results[0].__str__())

pipeline = RandomPipeline()
pipeline.register('resize', 'scale', 'rotate', 'deform', 'crop')
results = pipeline.process(image, label, mask)
imshow((image, label, mask), results)
plt.gcf().suptitle(results[0].__str__())

pipeline = RandomPipeline()
pipeline.register('flip', 'scale', 'rotate', 'deform', 'crop')
results = pipeline.process(image, label, mask)
imshow((image, label, mask), results)
plt.gcf().suptitle(results[0].__str__())

plt.show()
