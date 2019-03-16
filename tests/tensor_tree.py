#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from dataset.loads import load_label_hierachy, Hierachy
from dataset.tensor_tree import TensorLeaf, TensorTree


hierachy = load_label_hierachy('data/hierachy.json')
hierachy = hierachy.children[1].children[1].children[1]
data_shape = (1, 20, 160, 96, 96)

def create_sample(data_shape, hierachy, level=0):
    name = hierachy.name
    if isinstance(hierachy, Hierachy):
        data = np.random.rand(*data_shape).astype(np.int64)
        subtrees = list()
        for region in hierachy.children:
            subtrees.append(create_sample(data_shape, region, level=level+1))
        tensor_leaf = TensorTree(name, data, subtrees, level=level)
    else:
        tensor_leaf = TensorLeaf(name, level=level)
    return tensor_leaf

print(hierachy)
tensor_tree = create_sample(data_shape, hierachy)
print(tensor_tree)
