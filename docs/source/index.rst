Dataset
=======

This package provides a class :class:`dataset.Dataset` to iterate through image
data for deep network training and validation. It can apply various
pre-/post-processing and data augmentation and the augmentation is sampled
*randomly on the fly*. The supported operations include,

**Pre/post-processing**

* **Flip**: For example, flipping a brain image left and right can be regarded as
  creating new samples since the brain is approximately symmetric.
* **Crop**: The images can be cropped by a ROI mask to reduce data size.
* **Normalize a label image**: The label values are mapped to 0--(the number of
  unique labels - 1) to calcualte the loss.
* **Extract a mask image**: Extract a mask image from a label image corresponding
  to a pre-defined label value.
* **Extract patches**: Extract image patches (sub-regions). UNDER DEVELOPMENT.

**Augmentation**

* **Translate**: Translate an image in integer voxels.
* **Rotate**: Rotate an image around the x, y, and z axes.
* **Scaling**: Scale an image along x, y, and z axes.
* **Deformation**: Apply random elastic deformation to an image. The
  transformation field is a spatially smoothed per-voxel translation.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   usage
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
