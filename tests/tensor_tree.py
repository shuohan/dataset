#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from dataset.loads import load_tree
from dataset.trees import TensorLeaf, TensorTree, Tree


tree = load_tree('data/hierachy.json')
data_shape = (2, 80, 32, 16)

def create_sample(data_shape, tree, level=0):
    subtrees = dict()
    data = np.random.rand(*data_shape).astype(np.int64)
    for name, sub in tree.subtrees.items():
        data = np.random.rand(*data_shape).astype(np.int64)
        if isinstance(sub, Tree):
            subtrees[name] = create_sample(data_shape, sub, level=level+1)
        else:
            subtrees[name] = TensorLeaf(data, level=level+1)
    return TensorTree(subtrees, data, level=level)

print(tree)
tensor_tree = create_sample(data_shape, tree)
print(tensor_tree)
