# -*- coding: utf-8 -*-


class TensorLeaf:
    """

    Attributes:
        name (str): The identifier
        data (torch.Tensor): The tensor data
        print_level (int): The level of printing

        self._indent_pattern (str): The indentation pattern to print

    """
    def __init__(self, name, level=0):
        self.name = name
        self.level = level
        self._indent_pattern = '|   '
        self._name_prefix = '- '

    def __str__(self):
        indent = self._get_indent()
        name = self._name_prefix + self.name
        return '%s%s' % (indent, name) 

    def _get_indent(self):
        return self._indent_pattern * self.level


class TensorTree(TensorLeaf):
    """

    Attributes:
        subtrees (list of TensorLeaf): The children tensors

    """
    def __init__(self, name, data, subtrees, level=0):
        super().__init__(name)
        self.data = data
        self.level = level
        self.subtrees = subtrees

    def __str__(self):
        results = list()
        results.append(super().__str__() + self._desc_data())
        for subtree in self.subtrees:
            results.append(subtree.__str__())
        return '\n'.join(results)

    def _desc_data(self):
        dataid = id(self.data)
        dtype = self.data.type()
        shape = self.data.shape.__str__()
        return ': %d, %s, %s' % (dataid, dtype, shape)
