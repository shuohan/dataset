#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from dataset import Config
from dataset.images import Image, Label, Mask, LabelInfo, FileInfo
from dataset.workers import WorkerCreator

from plot import imshow


image1_filepath = 'data/at1000_image.nii.gz'
image2_filepath = 'data/at1006_image.nii.gz'
label_filepath = 'data/at1000_label.nii.gz'
mask_filepath = 'data/at1000_mask.nii.gz'
label_info = LabelInfo(filepath='data/labels.json')

image1 = Image(FileInfo(image1_filepath), on_the_fly=False)
image2 = Image(FileInfo(image2_filepath), on_the_fly=False)
label = Label(FileInfo(label_filepath), on_the_fly=False, label_info=label_info)
mask = Mask(FileInfo(mask_filepath), on_the_fly=False)
images = (image1, image2, label, mask)

creator = WorkerCreator()
print(creator)

# resize
resizer = creator.create('resize')
results = resizer.process(*images)
imshow(images, results)
plt.gcf().suptitle('resize')

# cropper
cropper = creator.create('crop')
cropped = cropper.process(*images)
imshow(images, cropped)
plt.gcf().suptitle('crop')

# rotate
rotator = creator.create('rotate')
results = cropper.process(*rotator.process(*images))
print(np.unique(results[2].data))
imshow(cropped, results)
plt.gcf().suptitle('rotate')

# deform
Config.def_scale = 10
deformer = creator.create('deform')
results = cropper.process(*deformer.process(*images))
imshow(cropped, results)
plt.gcf().suptitle('deform')

# scale
scaler = creator.create('scale')
results = cropper.process(*scaler.process(*images))
imshow(cropped, results)
plt.gcf().suptitle('scale')

# flip
flipper = creator.create('flip')
results = cropper.process(*flipper.process(*images))
imshow(cropped, results)
plt.gcf().suptitle('flip')

# translate
translator = creator.create('translate')
results = translator.process(*images)
imshow(images, results)
plt.gcf().suptitle('translate')

# extract mask
Config.mask_label_val = 12
extractor = creator.create('extract_mask')
results = cropper.process(*extractor.process(*images))
imshow(cropped, results)
plt.gcf().suptitle('extract mask')

# extract patches
Config.patch_shape = [100, 100, 100]
Config.num_patches = 3
extractor = creator.create('extract_patches')
results = extractor.process(*images)
imshow(results)
plt.gcf().suptitle('extract patches')

# norm label image
norm = creator.create('norm_label')
results = cropper.process(*norm.process(*images))
imshow(cropped, results)
plt.gcf().suptitle('norm labels')

plt.show()
