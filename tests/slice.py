#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataset import Config
from dataset.workers import SliceExtractor, Cropper
from dataset.funcs import patch_collate
from dataset.images import FileInfo, Image, Label, Mask
import matplotlib.pyplot as plt


num_slices1 = 5
Config.num_slices = num_slices1
worker = SliceExtractor()
cropper = Cropper()

image_info = FileInfo('dataset1/at1000_image.nii.gz')
label_info = FileInfo('dataset1/at1000_label.nii.gz')
mask_info = FileInfo('dataset1/at1000_mask.nii.gz')
images = (Image(info=image_info, on_the_fly=False),
          Label(info=label_info, on_the_fly=False),
          Mask(info=mask_info, on_the_fly=False))
slices1 = worker.process(*cropper.process(*images))

num_slices2 = 5
Config.num_slices = num_slices2
worker = SliceExtractor()
cropper = Cropper()

image_info = FileInfo('dataset1/at1006_image.nii.gz')
label_info = FileInfo('dataset1/at1006_label.nii.gz')
mask_info = FileInfo('dataset1/at1006_mask.nii.gz')
images = (Image(info=image_info, on_the_fly=False),
          Label(info=label_info, on_the_fly=False),
          Mask(info=mask_info, on_the_fly=False))
slices2 = worker.process(*cropper.process(*images))

slices = patch_collate(([s.data for s in slices1], [s.data for s in slices2]))
print(slices[0].shape)

for num in range(num_slices1 + num_slices2):
    plt.figure()
    image = slices[0][num].squeeze()
    label = slices[1][num].squeeze()
    plt.imshow(image, cmap='gray')
    plt.imshow(label, alpha=0.5)
plt.show()
