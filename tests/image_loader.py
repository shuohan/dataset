#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataset.images import ImageLoader, LabelLoader, MaskLoader
from dataset.images import BoundingBoxLoader, FileSearcher


fs = FileSearcher('data').search()
print(fs.files[0])

il = ImageLoader(fs).load()
ll = LabelLoader(fs).load()
ml = MaskLoader(fs).load()
bl = BoundingBoxLoader(fs).load()

# print(ml.images)
images = [il.images, ll.images, ml.images, bl.images]
images = sum(images[1:], images[0])
for k, v in images.items():
    print('----')
    print(k)
    for vv in v:
        print(vv)
