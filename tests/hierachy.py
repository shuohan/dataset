#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from dataset import DatasetFactory, Config

Config().dataset_type = 'wrapper_dataset'
t_ops = ('cropping', 'label_normalization')
v_ops = ('cropping', 'label_normalization')

# val_ind
factory = DatasetFactory()
factory.add_image_type('image', 'hierachical_label', 'mask', 'bounding_box')
factory.add_dataset(dataset_id='1', dirname='t_data_1', val_ind=[0])
factory.add_training_operation(*t_ops)
factory.add_validation_operation(*v_ops)
t_dataset, v_dataset = factory.create()
t_keys = ['1/at1006', '1/at1007']
v_keys = ['1/at1000']
assert list(t_dataset.images.keys()) == t_keys
assert list(v_dataset.images.keys()) == v_keys

image = t_dataset[0][1]
print(image)
print(image.region_tree)

tensor_tree = image.get_tensor_tree()
print(tensor_tree)
