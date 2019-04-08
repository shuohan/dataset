#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataset import DatasetFactory, Config
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.morphology import binary_dilation


Config().verbose = True
Config().mask_label_val = 63
Config().image_shape = (384, 384, 256)
t_ops = ('resizing', 'mask_extraction')
v_ops = ('resizing', 'mask_extraction')

# val_ind
factory = DatasetFactory()
factory.add_image_type('image', 'label')
factory.add_dataset(dataset_id='1', dirname='chaos_data')
factory.add_training_operation(*t_ops)
factory.add_validation_operation(*v_ops)
t_dataset, v_dataset = factory.create()
print(t_dataset)

# show image
def show(image, label):
    shape = image.shape[1:]
    fig = plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(image[0, shape[0]//2, :, :], cmap='gray')
    plt.imshow(label[0, shape[0]//2, :, :], alpha=0.3)
    plt.subplot(1, 3, 2)
    plt.imshow(image[0, :, shape[1]//2, :], cmap='gray')
    plt.imshow(label[0, :, shape[1]//2, :], alpha=0.3)
    plt.subplot(1, 3, 3)
    plt.imshow(image[0, :, :, shape[2]//2], cmap='gray')
    plt.imshow(label[0, :, :, shape[2]//2], alpha=0.3)

t1in_image, t1out_image, t2_image, t1_label, t2_label = t_dataset[10]
show(t1in_image, t1_label)
show(t1out_image, t1_label)
show(t2_image, t1_label)

show(t1in_image, t2_label)
show(t1out_image, t2_label)
show(t2_image, t2_label)

plt.show()
