#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataset.images import FileSearcher


fs = FileSearcher('data')
fs.search()
for f in fs.files:
    print('---')
    print(f)
print(fs.label_file)
