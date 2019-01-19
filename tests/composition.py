#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from scipy.ndimage.measurements import center_of_mass
import numpy as np
from memory_profiler import profile
from time import time

from network_utils.data import Data3d, Transforming3d
from network_utils.transformers import Rotater, Deformer, Cropper, Translater
from network_utils.transformers import LabelImageBinarizer, Scaler, Flipper


image_path = 'data/at1000_image.nii.gz'
label_path = 'data/at1000_label.nii.gz'
mask_path = 'data/at1000_mask.nii.gz'

label_pairs = [[33, 36], [43, 46], [53, 56], [63, 66], [73, 76], [74, 77],
               [75, 78], [83, 86], [84, 87], [93, 96], [103, 106]]
max_angle = 20
max_trans = 40
max_scale = 2
def_scale = 10
def_sigma = 5
cropping_shape=(128, 96, 96)

@profile
def test1():
    """ -> flipper -> rotater -> scaler -> deformer -> cropper -> binarizer"""
    image = Data3d(image_path, on_the_fly=False)
    label = Data3d(label_path, on_the_fly=False)
    mask = Data3d(mask_path, on_the_fly=False)

    flipper = Flipper()
    fimage = Transforming3d(image, flipper, on_the_fly=False)
    flabel = Transforming3d(label, flipper, on_the_fly=False,
                            label_pairs=label_pairs)
    fmask = Transforming3d(mask, flipper, on_the_fly=False)

    point = np.array(center_of_mass(np.squeeze(mask.get_data())))
    rotater = Rotater(max_angle=20, point=point)
    rimage = Transforming3d(fimage, rotater, on_the_fly=True)
    rlabel = Transforming3d(flabel, rotater, on_the_fly=True, order=0)
    rmask = Transforming3d(fmask, rotater, on_the_fly=True, order=0)

    deformer = Deformer(image.shape, def_sigma, def_scale)
    dimage = Transforming3d(rimage, deformer, on_the_fly=True)
    dlabel = Transforming3d(rlabel, deformer, on_the_fly=True, order=0)
    dmask = Transforming3d(rmask, deformer, on_the_fly=True, order=0)

    scaler = Scaler(max_scale=max_scale, point=point)
    simage = Transforming3d(dimage, scaler, on_the_fly=True)
    slabel = Transforming3d(dlabel, scaler, on_the_fly=True, order=0)
    smask = Transforming3d(dmask, scaler, on_the_fly=True, order=0)

    cropper = Cropper(smask, cropping_shape)
    cimage = Transforming3d(simage, cropper, on_the_fly=True)
    clabel = Transforming3d(slabel, cropper, on_the_fly=True)

    binarizer = LabelImageBinarizer()
    blabel = Transforming3d(clabel, binarizer, on_the_fly=True)

    images = [image, fimage, rimage, dimage, simage, cimage, cimage]
    labels = [label, flabel, rlabel, dlabel, slabel, clabel, blabel]
    num_cols = len(images)
    for i in range(2):
        plt.figure(figsize=(12, 6))
        print('Trial', i)
        blabel.update()
        for j, (im, la) in enumerate(zip(images, labels)):
            start_time = time()
            image_data = im.get_data()[0, ...]
            label_data = la.get_data()[0, ...]
            end_time = time()
            print('Time', j, end_time - start_time)
            shape = image_data.shape
            plt.subplot(3, num_cols, j + 1)
            plt.imshow(image_data[:, :, shape[2]//2], cmap='gray')
            plt.imshow(label_data[:, :, shape[2]//2], cmap='tab20', alpha=0.5)
            plt.subplot(3, num_cols, j + num_cols + 1)
            plt.imshow(image_data[:, shape[1]//2, :], cmap='gray')
            plt.imshow(label_data[:, shape[1]//2, :], cmap='tab20', alpha=0.5)
            plt.subplot(3, num_cols, j + 2 * num_cols + 1)
            plt.imshow(image_data[shape[0]//2, :, :], cmap='gray')
            plt.imshow(label_data[shape[0]//2, :, :], cmap='tab20', alpha=0.5)
        blabel.cleanup()

@profile
def test2():
    """ -> flipper -> rotater -> scaler -> deformer -> translater"""
    image = Data3d(image_path, on_the_fly=False)
    label = Data3d(label_path, on_the_fly=False)

    flipper = Flipper()
    fimage = Transforming3d(image, flipper, on_the_fly=False)
    flabel = Transforming3d(label, flipper, on_the_fly=False,
                            label_pairs=label_pairs)

    rotater = Rotater(max_angle=20, point=None)
    rimage = Transforming3d(fimage, rotater, on_the_fly=True)
    rlabel = Transforming3d(flabel, rotater, on_the_fly=True, order=0)

    deformer = Deformer(image.shape, def_sigma, def_scale)
    dimage = Transforming3d(rimage, deformer, on_the_fly=True)
    dlabel = Transforming3d(rlabel, deformer, on_the_fly=True, order=0)

    scaler = Scaler(max_scale=max_scale, point=None)
    simage = Transforming3d(dimage, scaler, on_the_fly=True)
    slabel = Transforming3d(dlabel, scaler, on_the_fly=True, order=0)

    translater = Translater(max_trans=max_trans)
    timage = Transforming3d(simage, translater, on_the_fly=True)
    tlabel = Transforming3d(slabel, translater, on_the_fly=True)

    images = [image, fimage, rimage, dimage, simage, timage]
    labels = [label, flabel, rlabel, dlabel, slabel, tlabel]
    num_cols = len(images)
    for i in range(2):
        plt.figure(figsize=(12, 6))
        print('Trial', i)
        tlabel.update()
        for j, (im, la) in enumerate(zip(images, labels)):
            start_time = time()
            image_data = im.get_data()[0, ...]
            label_data = la.get_data()[0, ...]
            end_time = time()
            print('Time', j, end_time - start_time)
            shape = image_data.shape
            plt.subplot(3, num_cols, j + 1)
            plt.imshow(image_data[:, :, shape[2]//2], cmap='gray')
            plt.imshow(label_data[:, :, shape[2]//2], cmap='tab20', alpha=0.5)
            plt.subplot(3, num_cols, j + num_cols + 1)
            plt.imshow(image_data[:, shape[1]//2, :], cmap='gray')
            plt.imshow(label_data[:, shape[1]//2, :], cmap='tab20', alpha=0.5)
            plt.subplot(3, num_cols, j + 2 * num_cols + 1)
            plt.imshow(image_data[shape[0]//2, :, :], cmap='gray')
            plt.imshow(label_data[shape[0]//2, :, :], cmap='tab20', alpha=0.5)
        tlabel.cleanup()
if __name__ == "__main__":
    test1()
    test2()
    plt.show()
