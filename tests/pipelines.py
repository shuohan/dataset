#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import numpy as np

from dataset import Config
from dataset.pipelines import RandomPipeline
from dataset.images import Image, Label, Mask, FileInfo, LabelInfo
from plot import imshow


image_filepath = 'dataset1/at1000_image.nii.gz'
label_filepath = 'dataset1/at1000_label.nii.gz'
mask_filepath = 'dataset1/at1000_mask.nii.gz'
label_desc_filepath = 'dataset1/labels.json'

image = Image(FileInfo(image_filepath), on_the_fly=False)
label = Label(FileInfo(label_filepath), on_the_fly=False,
              label_info=LabelInfo(label_desc_filepath))
mask = Mask(FileInfo(mask_filepath), on_the_fly=False)

Config.verbose = True

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
