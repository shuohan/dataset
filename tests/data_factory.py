#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')

import os
from glob import glob

from network_utils.data_factory import TrainingDataFactory, CroppedData3dFactory

image_paths = sorted(glob('data/*image.nii.gz'))
label_paths = sorted(glob('data/*label.nii.gz'))
mask_paths = sorted(glob('data/*mask.nii.gz'))

label_pairs = [[33, 36], [43, 46], [53, 56], [63, 66], [73, 76], [74, 77],
               [75, 78], [83, 86], [84, 87], [93, 96], [103, 106]]

factory = TrainingDataFactory(dim=0, label_pairs=label_pairs, max_angle=20)
factory = CroppedData3dFactory(factory, (128, 96, 96))

types = ['none', 'flipping', 'rotation', 'deformation']
for ip, lp, mp in zip(image_paths, label_paths, mask_paths):
    data = factory.create_data(ip, lp, mp, types=types)
    print(len(data))
    break
