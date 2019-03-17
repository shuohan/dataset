# -*- coding: utf-8 -*-

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
        self._indent_pattern = '|   '
        self._name_prefix = '- '

    def __str__(self):
        string = list()
        string.append('(#subtrees %d)' % len(self.subtrees))
        for name, subtree in self.subtrees.items():
            substring = list()
            substring = self._get_indent()
            substring += '%s%s' % (self._name_prefix, name)
            substring += ' ' + subtree.__str__()
            string.append(substring)
        return '\n'.join(string)

    def _get_indent(self):
        return self._indent_pattern * self.level
