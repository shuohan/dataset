#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from dataset import Config
from dataset.loads import load_label_desc
from dataset.images import Image, Label, Mask
from dataset.workers import WorkerCreator


def imshow(orig_images, processed_images):
    plt.figure()
    num_images = len(processed_images)
    for i, (oim, pim) in enumerate(zip(orig_images, processed_images)):
        if isinstance(oim, Label):
            cmap = 'gnuplot2'
            alpha = 0.3
            for im1, im2 in zip(orig_images, processed_images):
                if type(im1) is Image:
                    plt.subplot(2, num_images, num_images * 0 + i + 1)
                    plt.imshow(im1.data[0, :, :, im2.data.shape[-1]//2], cmap='gray')
                    plt.subplot(2, num_images, num_images * 1 + i + 1)
                    plt.imshow(im2.data[0, :, :, im2.data.shape[-1]//2], cmap='gray')
                    break
        else:
            cmap = 'gray'
            alpha = 1
        plt.subplot(2, num_images, num_images * 0 + i + 1)
        plt.imshow(oim.data[0, :, :, oim.data.shape[-1]//2], alpha=alpha, cmap=cmap)
        plt.subplot(2, num_images, num_images * 1 + i + 1)
        plt.imshow(pim.data[0, :, :, pim.data.shape[-1]//2], alpha=alpha, cmap=cmap)


image1_filepath = 'data/at1000_image.nii.gz'
image2_filepath = 'data/at1006_image.nii.gz'
label_filepath = 'data/at1000_label.nii.gz'
mask_filepath = 'data/at1000_mask.nii.gz'

image1 = Image(image1_filepath, on_the_fly=False)
image2 = Image(image2_filepath, on_the_fly=False)
label = Label(label_filepath, on_the_fly=False)
mask = Mask(mask_filepath, on_the_fly=False)
images = (image1, image2, label, mask)

labels, pairs = load_label_desc('data/labels.json')

creator = WorkerCreator()
print(creator)

# cropper
cropper = creator.create('crop')
cropped = cropper.process(*images)
# imshow(images, cropped)

# rotate
# rotator = creator.create('rotate')
# results = cropper.process(*rotator.process(*images))
# print(np.unique(results[2].data))
# imshow(cropped, results)

# deform
# config = Config()
# config.def_scale = 10
# deformer = creator.create('deform')
# results = cropper.process(*deformer.process(*images))
# imshow(cropped, results)

# scale
# scaler = creator.create('scale')
# results = cropper.process(*scaler.process(*images))
# imshow(cropped, results)

# flip
# flipper = creator.create('flip')
# results = cropper.process(*flipper.process(*images))
# imshow(cropped, results)

# translate
# translator = creator.create('translate')
# results = translator.process(*images)
# imshow(images, results)

# extract mask
# config = Config()
# config.mask_label_val = 12
# extractor = creator.create('extract_mask')
# results = cropper.process(*extractor.process(*images))
# imshow(cropped, results)

# extract patches
config = Config()
config.patch_shape = [100, 100, 100]
config.num_patches = 3
extractor = creator.create('extract_patches')
results = extractor.process(*images)
imshow(results, results)

# norm label image
norm = creator.create('norm_label')
results = cropper.process(*norm.process(*images))
imshow(cropped, results)

plt.show()
