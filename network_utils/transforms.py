# -*- coding: utf-8 -*-

import numpy as np
from functools import partial
from scipy.ndimage.measurements import center_of_mass

from image_processing_3d import rotate3d, deform3d, calc_random_deformation3d


def random_rotate3d(images, max_angle=5):
    orders = [1, 0, 0]
    point = None
    rand_state = np.random.RandomState()
    x_angle = float(rand_state.rand(1) * 2 * max_angle - max_angle)
    y_angle = float(rand_state.rand(1) * 2 * max_angle - max_angle)
    z_angle = float(rand_state.rand(1) * 2 * max_angle - max_angle)
    results = [rotate3d(im, x_angle, y_angle, z_angle, point=point, order=o)
               for im, o in zip(images, orders)]
    return results


# TODO: channel first 4D images
fliplr3d = np.flipud


def fliplr3d_label_image(label_image, label_pairs):
    label_image = fliplr3d(label_image)
    for (pair1, pair2) in label_pairs:
        mask1 = label_image == pair1
        mask2 = label_image == pair2
        label_image[mask1] = pair2
        label_image[mask2] = pair1
    return label_image
