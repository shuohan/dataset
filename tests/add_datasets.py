#!/usr/bin/env python
# -*- coding: utf-8 -*-


from dataset.datasets import DatasetCreator, Dataset
from dataset.pipelines import RandomPipeline


creator1 = DatasetCreator()
creator1.add_image_type('image')
creator1.add_dataset('dataset1')
dataset1 = creator1.create().dataset

creator2 = DatasetCreator()
creator2.add_image_type('image')
creator2.add_dataset('dataset2')
dataset2 = creator2.create().dataset


images = dataset1.images + dataset2.images
dataset = Dataset(images)

pipeline = RandomPipeline()
pipeline.register('resize', 'scale', 'rotate', 'crop')
dataset.add_pipeline(pipeline)

print('Dataset1')
print(dataset1)
print()

print('Dataset2')
print(dataset2)
print()

print('Total dataset')
print(dataset)
