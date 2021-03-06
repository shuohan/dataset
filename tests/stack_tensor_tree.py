#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from dataset import DatasetFactory, Config
from dataset.trees import TensorTree

# Config().dataset_type = 'wrapper_dataset'
t_ops = ('cropping', 'label_normalization')
v_ops = ('cropping', 'label_normalization')

factory = DatasetFactory()
factory.add_image_type('image', 'hierachical_label', 'mask')
factory.add_dataset(dataset_id='tmc', dirname='data')
factory.add_dataset(dataset_id='kki', dirname='ped_data')
factory.add_training_operation(*t_ops)
t_dataset, v_dataset = factory.create()

indices = [0, 1, len(t_dataset) - 1]
tensor_trees = [t_dataset[ind][1] for ind in indices]
print(tensor_trees[-1])
tensor_tree = TensorTree.stack(tensor_trees)
print(tensor_tree)
