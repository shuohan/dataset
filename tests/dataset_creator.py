#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataset.datasets import DatasetCreator


creator = DatasetCreator()
creator.add_image_type('image', 'label', 'mask')
creator.add_dataset('data', dataset_id='0')
creator.add_dataset('ped_data', dataset_id='1')
creator.add_operation('rotate', 'flip', 'scale', 'crop', 'norm_label')
creator.create()
print(creator)
