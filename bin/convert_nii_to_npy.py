#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import nibabel as nib
import numpy as np
import os

parser = argparse.ArgumentParser(description='Convert .nii or .nii.gz to .npy')
parser.add_argument('images', nargs='+',
                    help='The .nii/.nii.gz files to convert')
parser.add_argument('-o', '--output-dir', required=False, default='.',
                    help='The output direcotory')
args = parser.parse_args()

if not os.path.isdir(args.output_dir):
    raise IOError('%s does not exist' % args.output_dir)

for image_path in args.images:
    image = nib.load(image_path).get_data()
    basename = os.path.basename(image_path)
    if basename.endswith('.nii.gz'): 
        filename = basename.replace('.nii.gz', '')
    else:
        filename, _ = os.path.splitext(basename)
    output_path = os.path.join(args.output_dir, filename + '.npy')
    np.save(output_path, image)
