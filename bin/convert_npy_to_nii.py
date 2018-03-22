#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import nibabel as nib
import numpy as np
import os

parser = argparse.ArgumentParser(description='Convert .npy to .nii.gz/.nii')
parser.add_argument('images', nargs='+', help='.npy files to convert')
parser.add_argument('-r', '--reference', required=False, default=None,
                    help='Use the affine and header of the reference image; '
                         'if unspecified, use identity affine and blank header')
parser.add_argument('-o', '--output-dir', required=False, default='.',
                    help='The output direcotory')
parser.add_argement('-nz', '-not-compress', required=False, default=False,
                    action='store_trure', help='Do not compress to .nii.gz')
args = parser.parse_args()

if not os.path.isdir(args.output_dir):
    raise IOError('%s does not exist' % args.output_dir)

for image_path in args.images:
    image = np.load(image_path)
    if not args.reference:
        image_obj = nib.Nifti1Image(image, np.eye(4))
    else:
        ref_obj = nib.load(args.reference)
        image_obj = nib.Nifti1Image(image, ref_obj.affine, ref_obj.header)
    filename, ext = os.path.splitext(os.path.basename(image_path))
    if args.not_compress:
        ext = '.nii'
    else:
        ext = '.nii.gz'
    output_path = os.path.join(args.output_dir, filename+ext)
    image_obj.to_filename(output_path)
