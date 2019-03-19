# -*- coding: utf-8 -*-

INDENT_PATTERN = '|   '
NAME_PREFIX = '- '


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

    def __str__(self):
        return desc_data(self.data)


class TensorTree(Tree):
    def __init__(self, subtrees, data, level=0):
        super().__init__(subtrees, level)
        self.data = data

    @property
    def _tree_info(self):
        return desc_data(self.data)
