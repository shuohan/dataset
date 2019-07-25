#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from dataset.images import Label, Image


def imshow(*images):
    """Shows images

    Args:
        images (iterable): Elements are tuple of images.

    """
    plt.figure()
    num_ims = len(images)
    num_subs = len(images[0])

    for i, ims in enumerate(images):
        for j, im in enumerate(ims):
            slice_id = im.data.shape[-1] // 2
            if isinstance(im, Label):
                cmap = 'gnuplot2'
                alpha = 0.3
                for ref in ims:
                    if type(ref) is Image:
                        break
                plt.subplot(num_ims, num_subs, num_subs * i + j + 1)
                plt.imshow(ref.data[0, :, :, slice_id], cmap='gray')
            else:
                cmap = 'gray'
                alpha = 1
            plt.subplot(num_ims, num_subs, num_subs * i + j + 1)
            plt.imshow(im.data[0, :, :, slice_id], alpha=alpha, cmap=cmap)
