from math import ceil

import numpy as np
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.deit.deit_backbone import DeiTBackbone
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export("keras_hub.layers.DeiTImageConverter")
class DeiTImageConverter(ImageConverter):
    """Converts images to the format expected by a DeiT model.
    This layer performs image normalization using mean and standard deviation
    values.
    Args:
        crop_size: The tuple of integers. The target size for the cropped image.
        norm_mean: list or tuple of floats. Mean values for image normalization.
            Defaults to `[0.5, 0.5, 0.5]`.
        norm_std: list or tuple of floats. Standard deviation values for
            image normalization. Defaults to `[0.5, 0.5, 0.5]`.
        **kwargs: Additional keyword arguments passed to
            `keras_hub.layers.preprocessing.ImageConverter`.
    Examples:
    ```python
    import keras
    import numpy as np
    from keras_hub.src.models.deit.deit_image_converter import (
        DeiTImageConverter
    )
    # Example image (replace with your actual image data)
    image = np.random.rand(1, 384, 384, 3)  # Example: (B, H, W, C)
    # Create a DeiTImageConverter instance
    converter = DeiTImageConverter(
        image_size=(384, 384),
        scale=1/255.
    )
    # Preprocess the image
    preprocessed_image = converter(image)
    ```
    """

    backbone_cls = DeiTBackbone

    def __init__(
        self,
        crop_size=None,
        norm_mean=[0.5, 0.5, 0.5],
        norm_std=[0.5, 0.5, 0.5],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.crop_size = crop_size

    @preprocessing_function
    def call(self, inputs):
        # We perform the whole image preprocessing in child class as it is
        # customized and follows different order than the base class.
        if self.image_size is not None:
            inputs = self.resizing(inputs)
        # Allow dictionary input for handling bounding boxes.
        if isinstance(inputs, dict):
            x = inputs["images"]
        else:
            x = inputs
        if self.crop_size is not None:
            x = ops.convert_to_numpy(x)
            x = self._center_crop(x, self.crop_size)
        if self.scale is not None:
            # If we are scaling always cast to the compute dtype. We can't
            # leave things as an int type if we are scaling to [0, 1].
            scale = self._expand_non_channel_dims(self.scale, x)
            x, scale = self._convert_types(x, scale, self.compute_dtype)
            x = x * scale
        if self.offset is not None:
            offset = self._expand_non_channel_dims(self.offset, x)
            x, offset = self._convert_types(x, offset, x.dtype)
            x = x + offset
        # By default normalize using imagenet mean and std
        if self.norm_mean:
            x = x - self._expand_non_channel_dims(self.norm_mean, x)
        if self.norm_std:
            x = x / self._expand_non_channel_dims(self.norm_std, x)
        return x

    def _center_crop(self, image, size):
        """
        Crops the `image` to the specified `size` using a center crop.
        Note that if the image is too small to be cropped to the size given,
        it will be padded (so the returned result will always be of size
        `size`).

        Args:
            image: numpy array. The image to crop.
            size: The tuple of integers. The target size for the cropped image.

        Returns:
            `np.ndarray`: The cropped image.
        """

        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"Input image must be of type np.ndarray, got {type(image)}"
            )

        # We perform the crop in (C, H, W) format and then convert
        # to the output format
        if self.data_format == "channels_last":
            image = self._convert_data_format(image, "channels_first")

        orig_height, orig_width = image.shape[-2], image.shape[-1]
        crop_height, crop_width = size
        crop_height, crop_width = int(crop_height), int(crop_width)

        # In case size is odd, (image_shape[0] + size[0]) // 2
        # won't give the proper result.
        top = (orig_height - crop_height) // 2
        bottom = top + crop_height
        # In case size is odd, (image_shape[1] + size[1]) // 2
        # won't give the proper result.
        left = (orig_width - crop_width) // 2
        right = left + crop_width

        # Check if cropped area is within image boundaries
        if (
            top >= 0
            and bottom <= orig_height
            and left >= 0
            and right <= orig_width
        ):
            image = image[..., top:bottom, left:right]
            if self.data_format == "channels_last":
                image = self._convert_data_format(image, "channels_last")

            return image

        # Otherwise, we may need to pad if the image is too small. Oh joy...
        new_height = max(crop_height, orig_height)
        new_width = max(crop_width, orig_width)
        new_shape = image.shape[:-2] + (new_height, new_width)
        new_image = np.zeros_like(image, shape=new_shape)

        # If the image is too small, pad it with zeros
        top_pad = ceil((new_height - orig_height) / 2)
        bottom_pad = top_pad + orig_height
        left_pad = ceil((new_width - orig_width) / 2)
        right_pad = left_pad + orig_width
        new_image[..., top_pad:bottom_pad, left_pad:right_pad] = image

        top += top_pad
        bottom += top_pad
        left += left_pad
        right += left_pad

        new_image = new_image[
            ...,
            max(0, top) : min(new_height, bottom),
            max(0, left) : min(new_width, right),
        ]
        if self.data_format == "channels_last":
            new_image = self._convert_data_format(new_image, "channels_last")

        return new_image

    def _convert_data_format(self, image, to_format):
        """
        Convert image tensor between 'channels_first' and 'channels_last'.

        Args:
            image: numpy array. Image array of shape
                - (H, W, C) or (C, H, W) for unbatched
                - (B, H, W, C) or (B, C, H, W) for batched
            to_format: string. Target format
                - 'channels_first' or 'channels_last'

        Returns:
            np.ndarray: Transformed image tensor.
        """
        if to_format not in {"channels_first", "channels_last"}:
            raise ValueError(
                "`to_format` must be 'channels_first' or 'channels_last'."
            )

        ndim = image.ndim
        if to_format == "channels_first":
            if ndim == 4:
                return np.transpose(
                    image, (0, 3, 1, 2)
                )  # B, H, W, C → B, C, H, W
            elif ndim == 3:
                return np.transpose(image, (2, 0, 1))  # H, W, C → C, H, W
        else:  # to_format == "channels_last"
            if ndim == 4:
                return np.transpose(
                    image, (0, 2, 3, 1)
                )  # B, C, H, W → B, H, W, C
            elif ndim == 3:
                return np.transpose(image, (1, 2, 0))  # C, H, W → H, W, C

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "norm_mean": self.norm_mean,
                "norm_std": self.norm_std,
            }
        )
        return config
