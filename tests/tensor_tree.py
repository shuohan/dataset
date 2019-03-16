#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from dataset.loads import load_label_hierachy, Hierachy
from dataset.tensor_tree import TensorLeaf, TensorTree


hierachy = load_label_hierachy('data/hierachy.json')
hierachy = hierachy.children[1].children[1]
data_shape = (1, 20, 160, 96, 96)

def create_sample_tensor_tree(data_shape, hierachy):
    name = hierachy.name
    data = torch.rand(data_shape).float()
    if isinstance(hierachy, Hierachy):
        subtrees = list()
        for region in hierachy.children:
            subtrees.append(create_sample_tensor_tree(data_shape, region))
        tensor_leaf = TensorTree(name, data, subtrees)
    else:
        tensor_leaf = TensorLeaf(name, data)
    return tensor_leaf

print(hierachy)
tensor_tree = create_sample_tensor_tree(data_shape, hierachy)
print(tensor_tree)
