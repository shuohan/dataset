# Dataset

Dataset for 3D images with data augmentation and other operations. This implementation should be compatible with both PyTorch and Keras.

## Supported image types

Intensity images (e.g. MRI and CT), label images (delineation) and bounding box are supported.

## Supported Operations

The operations are divided into two types: augmentation and add-on. The augmentation is applied to the images on the fly with randomly sampled transformation. The add-on operations are some pre-processing and post-processing.

The following **augmentation** methods are supported:

* _Translation_
* _Rotation_
* _Scaling_: Scale the image alone x, y, and z axes.
* _Deformation_: Random elastic deformation. It is basically a spatially smoothed per-voxel translation.
* _Sigmoid intensity_ (**TODO**): Apply a mixture of sigmoid functions to perturb the image intensities.

The following **add-on** operations are supported:

* _Flipping_ (left/right): Flipping a brain image left and right can be regarded as creating new samples since the brain is symmetric.
* _Cropping_: The images can be cropped by a ROI mask to reduce data size.
* _Label normalizatoin_: The label values are normalized to 0 : number of unique labels.
* _Patch extraction_ (**TODO**): Extract image patch (sub-regions)

## Configurations

Check `dataset.configs.Config` for available configurations. **TODO**: load configurations from a `.json` file.

# Example

```python
from dataset import ImageLoader, Dataset, RandomPipeline

loader = ImageLoader(dirname, id='testing_dataset')
loader.load('image', 'label', 'mask', 'bounding_box') # mask for ROI cropping
images1, imagse2 = loader.split([0, 1, 2, 3])
dataset = Dataset(images1)

pipeline = RandomPipeline()
pipeline.register('scaling', 'rotation', 'cropping')
dataset.add_pipeline(pipeline)

print(dataset)
print(len(dataset))
print(dataset[0])
```
