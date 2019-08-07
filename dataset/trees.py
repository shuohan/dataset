# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict


INDENT_PATTERN = '|   '
NAME_PREFIX = '- '
STACK_DIM = 0


def load_tree(filepath):
    if os.path.isfile(filepath):
        with open(filepath) as jfile:
            tree = json.load(jfile)
        return _create_tree(tree)

def _create_tree(tree, level=0):
    if 'subregions' in tree:
        subtrees = {sub['name']: _create_tree(sub, level=level+1)
                    for sub in tree['subregions']}
        return Tree(subtrees, level=level)
    else:
        return Leaf(level=level)

class Leaf:
    """

    Attributes:
        level (int): Printing indentation level

    """
    def __init__(self, level=0):
        self.level = level

    def __str__(self):
        return ''


class Tree(Leaf):
    """

    Attributes:
        subtrees (dict): The subtrees
    
    """
    def __init__(self, subtrees, level=0):
        super().__init__(level)
        self.subtrees = subtrees

    def __str__(self):
        string = list()
        num_subtrees = len(self.subtrees)
        string.append('(#subtrees %d) %s' % (num_subtrees, self._tree_info))
        for name, subtree in self.subtrees.items():
            substring = list()
            substring = self._get_indent()
            substring += '%s%s' % (NAME_PREFIX, name)
            substring += ' ' + subtree.__str__()
            string.append(substring)
        return '\n'.join(string)

    def _get_indent(self):
        return INDENT_PATTERN * self.level

    @property
    def _tree_info(self):
        return ''


class RegionLeaf(Leaf):
    def __init__(self, value, level=0):
        super().__init__(level)
        self._value = value

    @property
    def value(self):
        return [self._value]

    def __str__(self):
        return '[%d]' % self._value


class RegionTree(Tree):

    @property
    def value(self):
        value = list()
        for tree in self.subtrees.values():
            value.extend(tree.value)
        return value

    @property
    def _tree_info(self):
        return '[%s]' % ', '.join([str(v) for v in self.value])


def desc_data(data):
    dtype = data.dtype.__str__()
    shape = data.shape.__str__()
    return '%s %s' % (dtype, shape)


class TensorLeaf(Leaf):

    def __init__(self, data, level=0):
        super().__init__(level)
        self.data = data
        self._desc_func = desc_data

    @property
    def desc_func(self):
        return self._desc_func

    @desc_func.setter
    def desc_func(self, func):
        self._desc_func = func

    def exec_data_attr(self, attr):
        self.data = getattr(self.data, attr)()
        return self

    def apply_to_data(self, func, *args, **kwargs):
        self.data = func(self.data, *args, **kwargs)
        return self

    def __str__(self):
        return self._desc_func(self.data)


class TensorTree(Tree):
    def __init__(self, subtrees, data, level=0):
        super().__init__(subtrees, level)
        self.data = data
        self._desc_func = desc_data
    
    @property
    def desc_func(self):
        return self._desc_func

    @desc_func.setter
    def desc_func(self, func):
        self._desc_func = func
        for subtree in self.subtrees.values():
            subtree.desc_func = func

    def exec_data_attr(self, attr):
        self.data = getattr(self.data, attr)()
        for subtree in self.subtrees.values():
            subtree.exec_data_attr(attr)
        return self

    def apply_to_data(self, func, *args, **kwargs):
        self.data = func(self.data, *args, **kwargs)
        for subtree in self.subtrees.values():
            subtree.apply_to_data(func, *args, **kwargs)
        return self

    @property
    def _tree_info(self):
        return self._desc_func(self.data)

    @classmethod
    def stack(cls, tensor_trees):
        """

        Args:
            tensor_trees (list of TensorTree): The tensor_tree to stack

        """
        level = tensor_trees[0].level
        for tree in tensor_trees:
            assert tree.level == level

        data = np.stack([tree.data for tree in tensor_trees], axis=STACK_DIM)

        subtrees = defaultdict(list)
        for tree in tensor_trees:
            if isinstance(tree, TensorTree):
                for name, subtree in tree.subtrees.items():
                    subtrees[name].append(subtree)

        if len(subtrees) > 0: # at least one subtree
            for name, trees in subtrees.items():
                subtrees[name] = TensorTree.stack(trees)
            return TensorTree(subtrees, data, level=level)
        else:
            return TensorLeaf(data, level=level)
