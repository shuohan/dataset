#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataset import DatasetFactory, Config

Config().verbose = True
Config().dataset_type = 'wrapper_dataset'
# t_ops = ('resizing', 'translation', 'scaling', 'deformation', 'rotation',
#          'sigmoid_intensity', 'cropping', 'label_normalization')
t_ops = ('cropping', 'label_normalization')
v_ops = ('cropping', 'label_normalization')

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
print(t_dataset[0][1].hierachy)
