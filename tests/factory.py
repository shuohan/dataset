#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataset import DatasetFactory, Config
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.morphology import binary_dilation


Config().verbose = True
t_ops = ('resize', 'translate', 'scale', 'deform', 'rotate', 'crop', 'norm_label')
v_ops = ('resize', 'crop', 'norm_label')

# val_ind
factory = DatasetFactory()
factory.add_image_type('image', 'label', 'mask', 'bounding_box')
factory.add_dataset(dataset_id='1', dirname='t_data_1', val_ind=[0, 2])
factory.add_dataset(dataset_id='2', dirname='t_data_2')
factory.add_dataset(dataset_id='3', dirname='t_data_3', val_ind=[0])
factory.add_training_operation(*t_ops)
factory.add_validation_operation(*v_ops)
t_dataset, v_dataset = factory.create()
t_keys = ['1/at1006', '2/at1025', '2/at1029', '3/at1034', '3/at1040']
v_keys = ['1/at1000', '1/at1007', '3/at1033']
assert list(t_dataset.images.keys()) == t_keys
assert list(v_dataset.images.keys()) == v_keys

# no val
factory = DatasetFactory()
factory.add_image_type('image', 'label', 'mask', 'bounding_box')
factory.add_dataset(dataset_id='1', dirname='t_data_1')
factory.add_training_operation(*t_ops)
factory.add_validation_operation(*v_ops)
t_dataset, v_dataset = factory.create()
t_keys = ['1/at1000', '1/at1006', '1/at1007']
assert list(t_dataset.images.keys()) == t_keys
assert len(v_dataset) == 0

# t_dirname v_dirname
factory = DatasetFactory()
factory.add_image_type('image', 'label', 'mask', 'bounding_box')
factory.add_dataset(dataset_id='1', t_dirname='t_data_1', v_dirname='v_data_1')
factory.add_dataset(dataset_id='2', t_dirname='t_data_2', v_dirname='v_data_2')
factory.add_dataset(dataset_id='3', t_dirname='t_data_3', v_dirname='v_data_3')
factory.add_training_operation(*t_ops)
factory.add_validation_operation(*v_ops)
t_dataset, v_dataset = factory.create()
t_keys = ['1/at1000', '1/at1006', '1/at1007', '2/at1025', '2/at1029',
          '3/at1033', '3/at1034', '3/at1040']
v_keys = ['1/at1017', '1/at1021', '1/at1031', '2/at1044', '2/at1048',
          '2/at1049', '3/at1084']
assert list(t_dataset.images.keys()) == t_keys
assert list(v_dataset.images.keys()) == v_keys

# show image
def show(image, label, bbox, title):
    bbox = bbox.astype(np.int)
    mask = np.zeros(image.shape, dtype=bool)
    mask[..., bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]] = 1
    mask = binary_dilation(mask) ^ mask
    shape = image.shape[1:]
    fig = plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(image[0, shape[0]//2, :, :], cmap='gray')
    plt.imshow(label[0, shape[0]//2, :, :], alpha=0.3)
    plt.imshow(mask[0, shape[0]//2, :, :], alpha=0.3, cmap='autumn')
    plt.subplot(1, 3, 2)
    plt.imshow(image[0, :, shape[1]//2, :], cmap='gray')
    plt.imshow(label[0, :, shape[1]//2, :], alpha=0.3)
    plt.imshow(mask[0, :, shape[1]//2, :], alpha=0.3, cmap='autumn')
    plt.subplot(1, 3, 3)
    plt.imshow(image[0, :, :, shape[2]//2], cmap='gray')
    plt.imshow(label[0, :, :, shape[2]//2], alpha=0.3)
    plt.imshow(mask[0, :, :, shape[2]//2], alpha=0.3, cmap='autumn')
    fig.suptitle(title)

t_image, t_label, t_bbox = t_dataset[0]
show(t_image, t_label, t_bbox, 'training')
v_image, v_label, v_bbox = v_dataset[0]
show(v_image, v_label, v_bbox, 'validation')

plt.show()
