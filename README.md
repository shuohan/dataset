# Dataset

Dataset for 3D images with data augmentation and other pre/post-processing. This implementation should be compatible with both PyTorch and Keras.

## Supported image types

Intensity images (e.g. MRI and CT), label images (delineation) and bounding box are supported. The images should be stored as the [NIfTI](https://nifti.nimh.nih.gov) format.

## Supported Operations

The operations are divided into two types: augmentation and add-on. The augmentation is applied to the images on the fly with randomly sampled transformation. The add-on operations are some pre-processing and post-processing.

The following **augmentation** methods are supported:

* _Translation_: Translate the images in integers (voxels).
* _Rotation_: Rotate the image around the x, y, and z axes.
* _Scaling_: Scale the image along x, y, and z axes.
* _Deformation_: Random elastic deformation. The transformation field is a spatially smoothed per-voxel translation.
* _Sigmoid intensity_ : Apply a mixture of sigmoid functions to perturb the image intensities. Altough it seems does no postitive effects on the networks.

The following **add-on** operations are supported:

* _Flipping_: Flipping a brain image left and right can be regarded as creating new samples since the brain is approximately symmetric.
* _Cropping_: The images can be cropped by a ROI mask to reduce data size.
* _Label normalizatoin_: The label values are mapped to 0--(the number of unique labels - 1).
* _Mask extraction_: Extract a mask image from a label image corresponding to a pre-defined label value.
* _Patch extraction_ : Extract image patches (sub-regions). UNDER DEVELOPMENT. **TODO**: Add pytorch `collate_fn` for `torch.utils.data.DataLoader` when extracting patches.

## Configurations

All the settings for augmentation and other functions are stored in a single place for global usage. Check the [documentation](https://shan-deep-networks.gitlab.io/dataset/) for more details.
