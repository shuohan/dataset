#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataset.loads import load_tree

filepath = 'data/hierachy.json'
tree = load_tree(filepath)
print(tree)
