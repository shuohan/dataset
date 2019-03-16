#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from dataset.loads import load_label_desc
from dataset.images import Label, Mask


filepath = 'data/at1000_label.nii.gz'
labels, pairs = load_label_desc('data/labels.json')

label = Label(filepath, on_the_fly=False)
mask = Mask('data/at1000_mask.nii.gz')
label = mask.crop(label)
norm = label.normalize()
for i, j in zip(labels.values(), np.arange(len(labels.values()))):
    assert np.array_equal(label.data==i, norm.data==j)

label = Label(filepath, labels=labels, pairs=pairs, on_the_fly=False)
label = mask.crop(label)
norm = label.normalize()
print(norm)
print(np.unique(norm.data))
print(labels.values())
for i, j in zip(labels.values(), np.arange(len(labels.values()))):
    assert np.array_equal(label.data==i, norm.data==j)

print(norm.labels)
sorted_labels = sorted(labels.items(), key=lambda l: l[1])
for i, (key, value) in enumerate(labels.items()):
    assert norm.labels[key] == i

for pair in norm.pairs:
    print(pair)

shape = norm.shape[1:]
plt.figure()
plt.subplot(2, 3, 1)
plt.imshow(norm.data[0, shape[0]//2, :, :])
plt.subplot(2, 3, 2)
plt.imshow(norm.data[0, :, shape[1]//2, :])
plt.subplot(2, 3, 3)
plt.imshow(norm.data[0, :, :, shape[2]//2])

plt.subplot(2, 3, 4)
plt.imshow(label.data[0, shape[0]//2, :, :])
plt.subplot(2, 3, 5)
plt.imshow(label.data[0, :, shape[1]//2, :])
plt.subplot(2, 3, 6)
plt.imshow(label.data[0, :, :, shape[2]//2])

plt.show()
