#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataset.datasets import DatasetCreator


creator = DatasetCreator()

# Add types of images to load
creator.add_image_type('image', 'label')
creator.add_image_type('mask')

# Add datasets stored in multiple directories
creator.add_dataset('dataset1', dataset_id='first')
creator.add_dataset('dataset2', dataset_id='second')

# Add operations; the adding orders are preserved
creator.add_operation('rotate', 'flip', 'scale')
creator.add_operation('crop', 'norm_label')

# Access the created dataset
dataset = creator.create().dataset

print('Dataset creator:')
print(creator)
print()

print('Raw images:')
for image in dataset.images.at(0):
    print(image)
print()

print('Processed images:')
for image in dataset.get_processed_image_group(0):
    print(image)
print()

print('Get data of processed images (shape):')
print([d.shape for d in dataset[0]])
