Usage
-----

File Structure
^^^^^^^^^^^^^^

To load a datasets, the images should be organized in the following way:

.. code-block::

   dataset_dirname/
      image1_image.nii.gz
      image1_label.nii.gz
      image1_mask.nii.gz
      image2_image.nii.gz
      image2_label.nii.gz
      image2_mask.nii.gz
      ...

The prefix in the file name (such as ``"image1"`` in ``"image1_image.nii.gz"``)
identifies the name of the subject, so the image (``"*image.nii.gz"``), the
label image (``"*label.nii.gz"``), and the mask (``"*mask.nii.gz"``, used to
crop other images) can be recognized as belonging to the same subject.

The suffixies (such as ``"label"`` in ``"image1_label.nii.gz"``) can be
configured to identify different images types,

>>> from dataset import Config
>>> Config.image_suffixes.append('T1w')
>>> Config.label_suffixes.append('delineation')
>>> Config.mask_suffixes.append('crop')

This package can also gather images from different directoires. See examples in
`Create a Dataset`_.

Operations
^^^^^^^^^^

The operations are categorized into :attr:`dataset.workers.WorkerType.ADDON` and
:attr:`dataset.workers.WorkerType.AUG` which correspond to pre/post-processing
and data augmentation, respectively. The types of operations/workers can be
configured in :attr:`dataset.Config.worker_types`.

Currently, after adding the operations into a dataset by calling
:meth:`dataset.DatasetCreator.add_operation`, all added ADDON operations are
applied to the images, but only one or none augmentation operation will be
randomly selected from the added AUG operations.


Create a Dataset
^^^^^^^^^^^^^^^^

This package provides two ways to create a dataset.

1. Use :class:`dataset.DatasetCreator`
""""""""""""""""""""""""""""""""""""""

.. literalinclude:: ../../tests/dataset_creator.py
   :language: python
   :linenos:

To check the available operations,

>>> from dataset.workers import WorkerCreator
>>> print(WorkerCreator())

or check the configuration :attr:`dataset.Config.worker_types` in the class
:class:`dataset.Config`.

2. Use lower-level APIs
"""""""""""""""""""""""

After creating an instance of :class:`dataset.dataset.Dataset`, additional
processing pipelines can be added into the dataset.  In fact, the length of a
dataset is equal to the number images times the number pipelines.

.. literalinclude:: ../../tests/add_pipeline.py
   :language: python
   :linenos:

Datasets can also be added together, though in a more complicated way,

.. literalinclude:: ../../tests/add_datasets.py
   :language: python
   :linenos:

Customize
^^^^^^^^^

Users can add more types of images via inheriting the class
:class:`dataset.images.Loader` and register the new classes into
:class:`dataset.DatasetCreator` by calling
:meth:`dataset.DatasetCreator.register_loader`.

Users can also implement new wokers by inheriting the class
:class:`dataset.workers.Worker`, specify the type (AUG or ADDON) of the worker
in :attr:`dataset.Config.worker_types`, and register them into
:class:`dataset.workers.WorkerCreator` by calling
:meth:`dataset.workers.WorkerCreator.register`.
