# Copyright 2024 The KerasHub Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export("keras_hub.layers.ResizingImageConverter")
class ResizingImageConverter(ImageConverter):
    """An `ImageConverter` that simply resizes the input image.

    The `ResizingImageConverter` is a subclass of `ImageConverter` for models
    that simply need to resize image tensors before using them for modeling.
    The layer will take as input a raw image tensor (batched or unbatched) in the
    channels last or channels first format, and output a resize tensor.

    Args:
        height: Integer, the height of the output shape.
        width: Integer, the width of the output shape.
        crop_to_aspect_ratio: If `True`, resize the images without aspect
            ratio distortion. When the original aspect ratio differs
            from the target aspect ratio, the output image will be
            cropped so as to return the
            largest possible window in the image (of size `(height, width)`)
            that matches the target aspect ratio. By default
            (`crop_to_aspect_ratio=False`), aspect ratio may not be preserved.
        interpolation: String, the interpolation method.
            Supports `"bilinear"`, `"nearest"`, `"bicubic"`,
            `"lanczos3"`, `"lanczos5"`. Defaults to `"bilinear"`.
        data_format: String, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.

    Examples:
    ```python
    # Resize images for `"pali_gemma_3b_224"`.
    converter = keras_hub.layers.ImageConverter.from_preset("pali_gemma_3b_224")
    converter(np.ones(2, 512, 512, 3)) # Output shape: (2, 224, 224, 3)
    # Resize images for `"pali_gemma_3b_224"`.
    converter = keras_hub.layers.ImageConverter.from_preset("pali_gemma_3b_448")
    converter(np.ones(2, 512, 512, 3)) # Output shape: (2, 448, 448, 3)
    ```
    """

    def __init__(
        self,
        height,
        width,
        crop_to_aspect_ratio=True,
        interpolation="bilinear",
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # By default, we just do a simple resize. Any model can subclass this
        # layer for preprocessing of a raw image to a model image input.
        self.resizing = keras.layers.Resizing(
            height=height,
            width=width,
            crop_to_aspect_ratio=crop_to_aspect_ratio,
            interpolation=interpolation,
            data_format=data_format,
        )

    def image_size(self):
        """Returns the preprocessed size of a single image."""
        return (self.resizing.height, self.resizing.width)

    @preprocessing_function
    def call(self, inputs):
        return self.resizing(inputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "height": self.resizing.height,
                "width": self.resizing.width,
                "interpolation": self.resizing.interpolation,
                "crop_to_aspect_ratio": self.resizing.crop_to_aspect_ratio,
            }
        )
        return config
