#!/usr/bin/env python
# -*- coding: utf-8 -*-

from network_utils.pipelines import RandomPipeline
from network_utils.datasets import Dataset, Delineated, Masked

dirname = 'data'
dataset = Dataset(verbose=True)
dataset = Delineated(dataset)
dataset = Masked(dataset)
dataset.add_images(dirname, id='tmc')
print(dataset)

from network_utils.workers import WorkerTypeMapping

worker_types = WorkerTypeMapping()
for key, value in worker_types.items():
    print(key, value)

pipeline = RandomPipeline()
pipeline.register('flipping')
pipeline.register('rotation')
pipeline.register('scaling')
pipeline.register('cropping')
dataset.add_pipeline(pipeline)

print('Length of dataset:', len(dataset))
images = dataset[5]
