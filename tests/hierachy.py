#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import nibabel as nib
from image_processing_3d import calc_bbox3d, resize_bbox3d, crop3d

from dataset import DatasetFactory, Config
from dataset.trees import Tree, TensorTree


ref_obj = nib.load('data/at1000_label.nii.gz')
# Config().dataset_type = 'wrapper_dataset'
Config().verbose = True
t_ops = ('cropping', 'label_normalization')
v_ops = ('cropping', 'label_normalization')
# t_ops = ('cropping', )
# v_ops = ('cropping', )

factory = DatasetFactory()
factory.add_image_type('image', 'hierachical_label', 'mask')
factory.add_dataset(dataset_id='tmc', dirname='data')
factory.add_dataset(dataset_id='kki', dirname='ped_data')
factory.add_training_operation(*t_ops)
factory.add_validation_operation(*v_ops)
t_dataset, v_dataset = factory.create()

mapping1 = {
    'Anterior Lobe': [33, 36, 43, 46, 53, 56],
    'Background': [0],
    'Cerebellum': [12, 33, 36, 43, 46, 53, 56, 60, 63, 66, 70, 73, 74, 75, 76,
                   77, 78, 80, 83, 84, 86, 87, 90, 93, 96, 103, 106, 100],
    'Corpus Medullare': [12],
    'Gray Matter': [33, 36, 43, 46, 53, 56, 60, 63, 66, 70, 73, 74, 75, 76,
                   77, 78, 80, 83, 84, 86, 87, 90, 93, 96, 103, 106, 100],
    'Inferior Posterior Lobe': [80, 83, 84, 86, 87, 90, 93, 96, 103, 106, 100],
    'Left Anterior Lobe': [33, 43, 53],
    'Left Crus I': [73],
    'Left Crus II': [74],
    'Left Crus II / VIIB': [74, 75],
    'Left I-III': [33],
    'Left IV': [43],
    'Left IX': [93],
    'Left Inferior Posterior Lobe': [83, 84, 93, 103],
    'Left Superior Posterior Lobe': [63, 73, 74, 75],
    'Left V': [53],
    'Left VI': [63],
    'Left VIIB': [75],
    'Left VIII': [83, 84],
    'Left VIIIA': [83],
    'Left VIIIB': [84],
    'Left X': [103],
    'Right Anterior Lobe': [36, 46, 56],
    'Right Crus I': [76],
    'Right Crus II': [77],
    'Right Crus II / VIIB': [77, 78],
    'Right I-III': [36],
    'Right IV': [46],
    'Right IX': [96],
    'Right Inferior Posterior Lobe': [86, 87, 96, 106],
    'Right Superior Posterior Lobe': [66, 76, 77, 78],
    'Right V': [56],
    'Right VI': [66],
    'Right VIIB': [78],
    'Right VIII': [86, 87],
    'Right VIIIA': [86],
    'Right VIIIB': [87],
    'Right X': [106],
    'Superior Posterior Lobe': [60, 63, 66, 70, 73, 74, 75, 76, 77, 78],
    'Vermis IX': [90],
    'Vermis Inferior Posterior': [80, 90, 100],
    'Vermis Superior Posterior': [60, 70],
    'Vermis VI': [60],
    'Vermis VII': [70],
    'Vermis VIII': [80],
    'Vermis X': [100]
}

mapping2 = {
    'Anterior Lobe': [3, 10, 17],
    'Background': [0],
    'Cerebellum': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'Corpus Medullare': [20],
    'Gray Matter': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'Inferior Posterior Lobe': [7, 8, 9, 14, 15, 16, 19],
    'Left Anterior Lobe': [10],
    'Left Crus I': [12],
    'Left Crus II / VIIB': [13],
    'Left IX': [15],
    'Left Inferior Posterior Lobe': [14, 15, 16],
    'Left Superior Posterior Lobe': [11, 12, 13],
    'Left VI': [11],
    'Left VIII': [14],
    'Left X': [16],
    'Right Anterior Lobe': [3],
    'Right Crus I': [5],
    'Right Crus II / VIIB': [6],
    'Right IX': [8],
    'Right Inferior Posterior Lobe': [7, 8, 9],
    'Right Superior Posterior Lobe': [4, 5, 6],
    'Right VI': [4],
    'Right VIII': [7],
    'Right X': [9],
    'Superior Posterior Lobe': [4, 5, 6, 11, 12, 13, 18],
    'Vermis Inferior Posterior': [19],
    'Vermis Superior Posterior': [18],
    'Vermis Anterior': [17]
}

image1, label1 = t_dataset[0]
image2, label2 = t_dataset[len(t_dataset)-1]

def get_ref(image_path, label_path, mask_path):
    image = nib.load(image_path).get_data()
    label = nib.load(label_path).get_data()
    mask = nib.load(mask_path).get_data()
    bbox = resize_bbox3d(calc_bbox3d(mask), (160, 96, 96))
    image = crop3d(image, bbox)[0]
    label = crop3d(label, bbox)[0]
    return image, label

def check(s_image_tree, s_label_tree, ref_images, ref_labels, mappings,
          name=None):
    if name is not None:
        indices = s_image_tree.indices
        for i in range(len(indices)):
            print(name, i)
            ind = indices[i]
            ref_masks = [ref_labels[ind]==v for v in mappings[ind][name]]
            ref_mask = np.logical_or.reduce(ref_masks).astype(np.int64)
            ref_image = ref_images[ind]
            image = s_image_tree.data[i, ...].squeeze()
            label = s_label_tree.data[i, ...].squeeze()
            assert np.array_equal(image, ref_image)
            assert np.array_equal(label, ref_mask)
    if isinstance(s_image_tree, Tree):
        for name, sub_imt in s_image_tree.subtrees.items():
            sub_lat = s_label_tree.subtrees[name]
            check(sub_imt, sub_lat, ref_images, ref_labels, mappings, name)

ref_image1, ref_label1 = get_ref('data/at1000_image.nii.gz',
                                 'data/at1000_label.nii.gz',
                                 'data/at1000_mask.nii.gz')
ref_image2, ref_label2 = get_ref('ped_data/2873_image.nii.gz',
                                 'ped_data/2873_label.nii.gz',
                                 'ped_data/2873_mask.nii.gz')

s_label_tree = TensorTree.stack((label1, label2))
# check(s_image_tree, s_label_tree, (ref_image1, ref_image2),
#       (ref_label1, ref_label2), (mapping1, mapping2))
# 
# print(s_image_tree)
# print(s_label_tree)

# dirname = 'results'
# if not os.path.isdir(dirname):
#     os.makedirs(dirname)
# 
# image = TensorTree.stack((image1, image2))
# def save(tree, filename=os.path.join(dirname, 'root')):
#     for i, data in enumerate(tree.data):
#         filepath = '%s_%d.nii.gz' % (filename, i)
#         print(filepath)
#         obj = nib.Nifti1Image(data.squeeze(), ref_obj.affine, ref_obj.header)
#         obj.to_filename(filepath)
#     if isinstance(tree, Tree):
#         for name, subtree in tree.subtrees.items():
#             name = name.replace(' ', '_').replace('/', '-')
#             filename = os.path.join(dirname, name)
#             save(subtree, filename=filename)
# save(image)
