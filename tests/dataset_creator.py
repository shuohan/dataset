#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataset.datasets import DatasetCreator


creator = DatasetCreator()
creator.add_image_type('image', 'label', 'mask')
creator.add_dataset('data', dataset_id='0')
creator.add_dataset('ped_data', dataset_id='1')
creator.add_operation('rotate', 'flip', 'scale', 'crop', 'norm_label')
dataset = creator.create().dataset

print('raw image')
for image in dataset.images.at(0):
    print(image)

print('processed image')
for image in dataset.get_processed_image_group(0):
    print(image)
