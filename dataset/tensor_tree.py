# -*- coding: utf-8 -*-


class TensorLeaf:
    """

    Attributes:
        name (str): The identifier
        data (torch.Tensor): The tensor data
        print_level (int): The level of printing

        self._indent_pattern (str): The indentation pattern to print

    """
    def __init__(self, name, data):
        self.name = name
        self.data = data
        self.print_level = 0
        self._indent_pattern = '|   '
        self._name_prefix = '- '

    def __str__(self):
        dataid = id(self.data)
        dtype = self.data.type()
        shape = self.data.shape.__str__()
        name = self._name_prefix + self.name
        indent = self._get_indent()
        return '%s%s: %d, %s, %s' % (indent, name, dataid, dtype, shape)

    def _get_indent(self):
        return self._indent_pattern * self.print_level


class TensorTree(TensorLeaf):
    """

    Attributes:
        subtrees (list of TensorLeaf): The children tensors

    """
    def __init__(self, name, data, subtrees):
        super().__init__(name, data)
        self.subtrees = subtrees

    def __str__(self):
        results = list()
        results.append(super().__str__())
        for subtree in self.subtrees:
            subtree.print_level = self.print_level + 1
            results.append(subtree.__str__())
        return '\n'.join(results)
