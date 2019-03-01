#!/usr/bin/env python
# -*- coding: utf-8 -*-

from network_utils.pipelines import SerialPipeline, RandomPipeline
from network_utils.datasets import Dataset, Delineated, Masked

dirname = 'data'
dataset = Dataset(verbose=True)
dataset = Delineated(dataset)
dataset = Masked(dataset)
dataset.add_images(dirname, id='tmc')
print(dataset)

pipeline = SerialPipeline()
pipeline.register('flipping')
pipeline.register('rotation')
pipeline.register('cropping')
dataset.add_pipeline(pipeline)

print('Length of dataset:', len(dataset))
images = dataset[5]

# 

dataset = Dataset(verbose=True)
dataset = Delineated(dataset)
dataset = Masked(dataset)
dataset.add_images(dirname, id='tmc')

pipeline = RandomPipeline()
pipeline.register('flipping')
pipeline.register('rotation')
pipeline.register('scaling')
pipeline.register('cropping')
print(pipeline.worker_priorities)
print(pipeline.random_worker_priorities)
dataset.add_pipeline(pipeline)

print('Length of dataset:', len(dataset))
images = dataset[5]
images = dataset[1]
images = dataset[10]

# 

dataset = Dataset(verbose=True)
dataset = Delineated(dataset)
dataset = Masked(dataset)
dataset.add_images(dirname, id='tmc')

pipeline = SerialPipeline()
pipeline.register('flipping')
pipeline.register('rotation')
pipeline.register('cropping')
dataset.add_pipeline(pipeline)

pipeline = SerialPipeline()
pipeline.register('flipping')
pipeline.register('scaling')
pipeline.register('cropping')
dataset.add_pipeline(pipeline)

print('Length of dataset:', len(dataset))
images = dataset[3]
images = dataset[24]
