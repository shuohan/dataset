#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataset.datasets import DatasetCreator
from dataset.pipelines import RandomPipeline


creator = DatasetCreator()
creator.add_image_type('image')
creator.add_operation('flip', 'scale', 'crop')
creator.add_dataset('dataset1')
dataset = creator.create().dataset
print('Before adding another pipeline, dataset length', len(dataset))

pipeline = RandomPipeline()
pipeline.register('resize', 'scale', 'rotate', 'crop')
dataset.add_pipeline(pipeline)
print('After adding another pipeline, dataset length', len(dataset))
