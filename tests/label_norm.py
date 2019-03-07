#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from network_utils.loads import load_label_desc
from network_utils.images import Label, Mask


filepath = 'data/at1000_label.nii.gz'
labels, pairs = load_label_desc('data/labels.json')

label = Label(filepath, on_the_fly=False)
mask = Mask('data/at1000_mask.nii.gz')
label = mask.crop(label)
norm = label.normalize()
print(norm)
for i, j in zip(labels.keys(), np.arange(len(labels.keys()))):
    assert np.array_equal(label.data==i, norm.data==j)

label = Label(filepath, labels=labels, pairs=pairs, on_the_fly=False)
label = mask.crop(label)
norm = label.normalize()
print(norm)
for i, j in zip(labels.keys(), np.arange(len(labels.keys()))):
    assert np.array_equal(label.data==i, norm.data==j)

shape = norm.shape
plt.figure()
plt.subplot(2, 3, 1)
plt.imshow(norm.data[shape[0]//2, :, :])
plt.subplot(2, 3, 2)
plt.imshow(norm.data[:, shape[1]//2, :])
plt.subplot(2, 3, 3)
plt.imshow(norm.data[:, :, shape[2]//2])

plt.subplot(2, 3, 4)
plt.imshow(label.data[shape[0]//2, :, :])
plt.subplot(2, 3, 5)
plt.imshow(label.data[:, shape[1]//2, :])
plt.subplot(2, 3, 6)
plt.imshow(label.data[:, :, shape[2]//2])

plt.show()
