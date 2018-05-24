#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')

from sys import getsizeof
import numpy as np
import matplotlib.pyplot as plt
from time import time
import tracemalloc
from memory_profiler import profile

from network_utils.data import Data3d
from network_utils.data_decorators import Cropping3d, Interpolating3d
from network_utils.data_decorators import Flipping3d
from network_utils.transformers import Deformer, Flipper


load_on_the_fly = True
trace = False

@profile
def main():

    if trace:
        tracemalloc.start()

    filepath = '/home/shuo/projects/nph/data/AQ002_image.nii.gz'
    data1 = Data3d(filepath, get_data_on_the_fly=load_on_the_fly)
    filepath = '/home/shuo/projects/nph/data/AQ002_mask.nii.gz'
    mask1 = Data3d(filepath, get_data_on_the_fly=load_on_the_fly)

    label_pairs = [[33, 36], [43, 46], [53, 56], [63, 66], [73, 76], [74, 77],
                   [75, 78], [83, 86], [84, 87], [93, 96], [103, 106]]
    flipper = Flipper(dim=1)

    if trace:
        snapshot1 = tracemalloc.take_snapshot()

    shape = data1.get_data().shape[-3:]

    if trace:
        snapshot2 = tracemalloc.take_snapshot()
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        for stat in top_stats[:10]:
            print(stat)
        print() 

    deformer = Deformer(shape, sigma=3, scale=8)

    data2 = Flipping3d(data1, flipper, get_data_on_the_fly=load_on_the_fly)
    data2 = Interpolating3d(data2, deformer, get_data_on_the_fly=True, order=1)
    mask2 = Flipping3d(mask1, flipper, get_data_on_the_fly=load_on_the_fly)
    mask2 = Interpolating3d(mask2, deformer, get_data_on_the_fly=True, order=0)

    data1 = Cropping3d(data1, mask1, (128, 96, 96),
                       get_data_on_the_fly=load_on_the_fly)
    data2 = Cropping3d(data2, mask2, (128, 96, 96),
                       get_data_on_the_fly=True)

    data1.update()
    if trace:
        snapshot1 = tracemalloc.take_snapshot()

    data2.update()
    if trace:
        snapshot2 = tracemalloc.take_snapshot()
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        for stat in top_stats[:10]:
            print(stat)
        print() 

    start_time = time()

    if trace:
        snapshot1 = tracemalloc.take_snapshot()

    print('get_data')
    data2.get_data()
    if load_on_the_fly:
        data2.cleanup()

    if trace:
        snapshot2 = tracemalloc.take_snapshot()
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        for stat in top_stats[:10]:
            print(stat)

    end_time = time()
    print('first load', end_time - start_time)

    data1.update()
    data2.update()

    start_time = time()
    data2.get_data()
    if load_on_the_fly:
        data2.cleanup()
    end_time = time()
    print('second load', end_time - start_time)

    data1.update()
    data2.update()

    start_time = time()
    data2.get_data()
    if load_on_the_fly:
        data2.cleanup()
    end_time = time()
    data1.update()
    data2.update()

    print('third load', end_time - start_time)

if __name__ == '__main__':
    main()
