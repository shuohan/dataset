#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataset.images import LabelInfo

info1 = LabelInfo('data/labels.json')
info2 = LabelInfo('data/labels.json')
info3 = LabelInfo('ped_data/labels.json')
info4 = LabelInfo('ped_data/labels.json')
info5 = LabelInfo('ped_data/labels.json')

print(info1.labels)
print(info3.labels)

print(hash(info1))
print(hash(info2))
print(hash(info3))

print(info1 == info2)
print(info1 is info2)

print(set((info1, info2, info3, info4, info5)))

a = {info1:1, info2:2}
print(a)
