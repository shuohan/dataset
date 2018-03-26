# -*- coding: utf-8 -*-

from sklearn.preprocessing import LabelBinarizer


class LabelImageBinarizer(LabelBinarizer):
    """Binarize label image (one-hot encoding)

    For example, if the label images is [1, 12; 3, 12] with 3 different labels.
    Binarization first map these labels to binary codes 1 -> [1, 0, 0],
    3 -> [0, 1, 0], 12 -> [0, 0, 1], and the label image is converted to
    [1, 0; 0, 0], [0, 0; 1, 0], and [0, 1; 0, 1] for these 3 binary channels.

    Check sklearn.preprocessing.LabelBinarizer for more details. The parent
    class can only transform 1D labels. This class can transform higher
    dimensional label image.

    Note that the output is channel LAST.

    """
    def fit(self, label_image):
        """Fit label image binarizer. Do not support sparse array.

        Args:
            label_image (numpy.array): The label image to fit. It can be 2D, 3D,
                ...

        Returns:
            self

        """
        return super().fit(label_image.flatten())

    def transform(self, label_image):
        """Transform label image to binary labels

        Args:
            label_image (numpy.array): The label image to transform. It can be
                2D, 3D, ...

        Returns:
            binarization (num_i x num_j x ... x num_channels numpy.array): The
                binary label image

        """
        flattened_label_image = label_image.flatten()
        binarization = super().transform(flattened_label_image)
        num_channels = len(self.classes_) if len(self.classes_) > 2 else 1
        binarization = binarization.reshape((*label_image.shape, num_channels))
        return binarization
    
    def inverse_transform(self, binarized_label_image):
        """Transform binary label image to multi-class label image
        
        Args:
            binarized_label_image (num_i x ... x num_channels numpy.array): The
                binary label image

        Returns:
            label_image (numpy.array): The multi-class label image

        """
        tmp = super().inverse_transform(binarized_label_image.flatten())
        label_image = tmp.reshape(binarized_label_image.shape[:-1])
        return label_image
