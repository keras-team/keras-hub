import math

import keras
import numpy as np
import tensorflow as tf

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.preprocessing_layer import (
    PreprocessingLayer,
)
from keras_hub.src.models.hgnetv2.hgnetv2_backbone import HGNetV2Backbone
from keras_hub.src.utils.keras_utils import standardize_data_format
from keras_hub.src.utils.preset_utils import builtin_presets
from keras_hub.src.utils.preset_utils import find_subclass
from keras_hub.src.utils.preset_utils import get_preset_loader
from keras_hub.src.utils.preset_utils import get_preset_saver
from keras_hub.src.utils.python_utils import classproperty
from keras_hub.src.utils.tensor_utils import in_tf_function
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras.saving.register_keras_serializable(package="keras_hub")
class ResizeThenCrop(keras.layers.Layer):
    """Resize and crop images to a target size while preserving aspect ratio.

    This layer resizes an input image to an intermediate size based on the
    `crop_pct` parameter, preserving the aspect ratio, and then performs a
    center crop to achieve the specified `target_size`. This preprocessing step
    is commonly used in image classification models to prepare inputs for
    neural networks.

    The preprocessing follows these steps:
    1. Compute an intermediate size by dividing the target height by `crop_pct`.
    2. Resize the image to the intermediate size, maintaining the aspect ratio.
    3. Crop the resized image from the center to the specified `target_size`.

    The layer accepts batched or unbatched image tensors with shape
    `(..., height, width, channels)` (`channels_last` format).

    Args:
        target_size: `(int, int)` tuple. The desired output size (height, width)
            of the image, excluding the channels dimension.
        crop_pct: float. The cropping percentage, typically between 0.0 and 1.0,
            used to compute the intermediate resize size. For example, a
            `crop_pct` of 0.875 means the intermediate height is
            `target_height / 0.875`.
        interpolation: String, the interpolation method for resizing.
            Supports `"bilinear"`, `"nearest"`, `"bicubic"`, `"lanczos3"`,
            `"lanczos5"`. Defaults to `"bilinear"`.
        **kwargs: Additional keyword arguments passed to the parent
            `keras.layers.Layer` class, such as `name` or `dtype`.
    """

    def __init__(
        self,
        target_size,
        crop_pct,
        interpolation="bilinear",
        antialias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_size = target_size
        self.crop_pct = crop_pct
        self.interpolation = interpolation
        self.antialias = antialias

    def call(self, inputs):
        target_height, target_width = self.target_size
        intermediate_size = int(math.floor(target_height / self.crop_pct))
        input_shape = keras.ops.shape(inputs)
        height = input_shape[-3]
        width = input_shape[-2]
        aspect_ratio = keras.ops.cast(width, "float32") / keras.ops.cast(
            height, "float32"
        )
        if keras.ops.is_tensor(aspect_ratio):
            resize_height = keras.ops.cond(
                aspect_ratio > 1,
                lambda: intermediate_size,
                lambda: keras.ops.cast(
                    intermediate_size / aspect_ratio, "int32"
                ),
            )
            resize_width = keras.ops.cond(
                aspect_ratio > 1,
                lambda: keras.ops.cast(
                    intermediate_size * aspect_ratio, "int32"
                ),
                lambda: intermediate_size,
            )
        else:
            if aspect_ratio > 1:
                resize_height = intermediate_size
                resize_width = int(intermediate_size * aspect_ratio)
            else:
                resize_width = intermediate_size
                resize_height = int(intermediate_size / aspect_ratio)
        resized = keras.ops.image.resize(
            inputs,
            (resize_height, resize_width),
            interpolation=self.interpolation,
            antialias=self.antialias,
        )
        top = (resize_height - target_height) // 2
        left = (resize_width - target_width) // 2
        cropped = resized[
            :, top : top + target_height, left : left + target_width, :
        ]
        return cropped

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "target_size": self.target_size,
                "crop_pct": self.crop_pct,
                "antialias": self.antialias,
                "interpolation": self.interpolation,
            }
        )
        return config


@keras_hub_export("keras_hub.layers.HGNetV2ImageConverter")
class HGNetV2ImageConverter(PreprocessingLayer):
    """Preprocess raw images into model-ready inputs for HGNetV2 models.

    This layer converts raw images into inputs suitable for HGNetV2 models.
    The preprocessing includes resizing, cropping, scaling, and normalization
    steps tailored for HGNetV2 architectures. The conversion proceeds in the
    following steps:

    1. Resize and crop the image to `image_size` using a `ResizeThenCrop` layer
       with the specified `crop_pct` if `image_size` is provided. If
       `image_size` is `None`, this step is skipped.
    2. Scale the image by dividing pixel values by 255.0 to normalize to [0, 1].
    3. Normalize the image by subtracting the `mean` and dividing by the `std`
       (per channel) if both are provided. If `mean` or `std` is `None`, this
       step is skipped.

    The layer accepts batched or unbatched image tensors in channels_last or
    channels_first format, with shape `(..., height, width, channels)` or
    `(..., channels, height, width)`, respectively. It can also handle
    dictionary inputs with an `"images"` key for compatibility with bounding box
    preprocessing.

    This layer can be instantiated using the `from_preset()` constructor to load
    preprocessing configurations for specific HGNetV2 presets, ensuring
    compatibility with pretrained models.

    Args:
        image_size: `(int, int)` tuple or `None`. The output size of the image
            (height, width), excluding the channels axis. If `None`, resizing
            and cropping are skipped.
        crop_pct: float. The cropping percentage used in the `ResizeThenCrop`
            layer to compute the intermediate resize size. Defaults to 0.875.
        mean: list or tuple of floats, or `None`. Per-channel mean values for
            normalization. If provided, these are subtracted from the image
            after scaling. If `None`, this step is skipped.
        std: list or tuple of floats, or `None`. Per-channel standard deviation
            values for normalization. If provided, the image is divided by these
            after mean subtraction. If `None`, this step is skipped.
        interpolation: String, the interpolation method for resizing.
            Supports `"bilinear"`, `"nearest"`, `"bicubic"`, `"lanczos3"`,
            `"lanczos5"`. Defaults to `"bilinear"`.
        antialias: Whether to use an antialiasing filter when downsampling an
            image. Defaults to `False`.
        bounding_box_format: A string specifying the format of the bounding
            boxes, one of `"xyxy"`, `"rel_xyxy"`, `"xywh"`, `"center_xywh"`,
            `"yxyx"`, `"rel_yxyx"`. Specifies the format of the bounding boxes
            which will be resized to `image_size` along with the image. To pass
            bounding boxes to this layer, pass a dict with keys `"images"` and
            `"bounding_boxes"` when calling the layer.
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
    import keras
    import numpy as np

    # Create an HGNetV2ImageConverter for a specific image size.
    converter = keras_hub.layers.HGNetV2ImageConverter(
        image_size=(224, 224),
        crop_pct=0.965,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        interpolation="bicubic",
    )
    images = np.random.randint(0, 256, size=(2, 512, 512, 3)).astype("float32")
    processed_images = converter(images)

    # Load an HGNetV2ImageConverter from a preset.
    converter = keras_hub.layers.HGNetV2ImageConverter.from_preset(
        "hgnetv2_b5.ssld_stage1_in22k_in1k"
    )
    processed_images = converter(images)
    """

    backbone_cls = HGNetV2Backbone

    def __init__(
        self,
        image_size=None,
        crop_pct=0.875,
        mean=None,
        std=None,
        scale=None,
        offset=None,
        crop_to_aspect_ratio=True,
        pad_to_aspect_ratio=False,
        interpolation="bilinear",
        antialias=False,
        bounding_box_format="yxyx",
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if crop_to_aspect_ratio and pad_to_aspect_ratio:
            raise ValueError(
                "Only one of 'crop_to_aspect_ratio' or 'pad_to_aspect_ratio' "
                "can be True."
            )

        self.image_size_val = image_size
        self.crop_pct = crop_pct
        self.mean = mean
        self.std = std
        self.scale = scale
        self.offset = offset
        self.crop_to_aspect_ratio = crop_to_aspect_ratio
        self.pad_to_aspect_ratio = pad_to_aspect_ratio
        self.interpolation = interpolation
        self.antialias = antialias
        self.bounding_box_format = bounding_box_format
        self.data_format = standardize_data_format(data_format)

        self.custom_resizing = None
        if image_size is not None:
            self.custom_resizing = ResizeThenCrop(
                target_size=image_size,
                crop_pct=crop_pct,
                interpolation=interpolation,
                antialias=antialias,
                dtype=self.dtype_policy,
                name="custom_resizing",
            )
        self.built = True

    @property
    def image_size(self):
        return self.image_size_val

    @image_size.setter
    def image_size(self, value):
        self.image_size_val = value
        if value is not None:
            self.custom_resizing = ResizeThenCrop(
                target_size=value,
                crop_pct=self.crop_pct,
                interpolation=self.interpolation,
                antialias=self.antialias,
                dtype=self.dtype_policy,
                name="custom_resizing",
            )
        else:
            self.custom_resizing = None

    @preprocessing_function
    def call(self, inputs):
        if self.image_size is not None and self.custom_resizing is not None:
            if in_tf_function():
                target_height, target_width = self.image_size
                intermediate_size = tf.cast(
                    tf.math.floor(target_height / self.crop_pct), tf.int32
                )
                input_shape = tf.shape(inputs)
                height = input_shape[-3]
                width = input_shape[-2]
                aspect_ratio = tf.cast(width, tf.float32) / tf.cast(
                    height, tf.float32
                )
                resize_height = tf.cond(
                    aspect_ratio > 1,
                    lambda: intermediate_size,
                    lambda: tf.cast(
                        tf.cast(intermediate_size, tf.float32) / aspect_ratio,
                        tf.int32,
                    ),
                )
                resize_width = tf.cond(
                    aspect_ratio > 1,
                    lambda: tf.cast(
                        tf.cast(intermediate_size, tf.float32) * aspect_ratio,
                        tf.int32,
                    ),
                    lambda: intermediate_size,
                )
                resized = tf.image.resize(
                    inputs,
                    [resize_height, resize_width],
                    method=self.interpolation,
                    antialias=self.antialias,
                )
                top = (resize_height - target_height) // 2
                left = (resize_width - target_width) // 2
                cropped = resized[
                    :, top : top + target_height, left : left + target_width, :
                ]
                inputs = cropped
            else:
                inputs = self.custom_resizing(inputs)
        if isinstance(inputs, dict):
            x = inputs["images"]
        else:
            x = inputs
        if in_tf_function():
            x = tf.cast(x, self.compute_dtype) / 255.0
        else:
            x = keras.ops.cast(x, self.compute_dtype) / 255.0
        if self.mean is not None and self.std is not None:
            mean = self._expand_non_channel_dims(self.mean, x)
            std = self._expand_non_channel_dims(self.std, x)
            x, mean = self._convert_types(x, mean, self.compute_dtype)
            x, std = self._convert_types(x, std, self.compute_dtype)
            x = (x - mean) / std
        if self.scale is not None:
            scale = self._expand_non_channel_dims(self.scale, x)
            x, scale = self._convert_types(x, scale, self.compute_dtype)
            x = x * scale
        if self.offset is not None:
            offset = self._expand_non_channel_dims(self.offset, x)
            x, offset = self._convert_types(x, offset, x.dtype)
            x = x + offset
        if isinstance(inputs, dict):
            inputs["images"] = x
        else:
            inputs = x
        return inputs

    def _expand_non_channel_dims(self, value, inputs):
        unbatched = len(keras.ops.shape(inputs)) == 3
        channels_first = self.data_format == "channels_first"
        if unbatched:
            broadcast_dims = (1, 2) if channels_first else (0, 1)
        else:
            broadcast_dims = (0, 2, 3) if channels_first else (0, 1, 2)
        return np.expand_dims(value, broadcast_dims)

    def _convert_types(self, x, y, dtype):
        if in_tf_function():
            return tf.cast(x, dtype), tf.cast(y, dtype)
        x = keras.ops.cast(x, dtype)
        y = keras.ops.cast(y, dtype)
        if keras.backend.backend() == "torch":
            y = y.to(x.device)
        return x, y

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_size": self.image_size,
                "crop_pct": self.crop_pct,
                "mean": self.mean,
                "std": self.std,
                "scale": self.scale,
                "offset": self.offset,
                "crop_to_aspect_ratio": self.crop_to_aspect_ratio,
                "pad_to_aspect_ratio": self.pad_to_aspect_ratio,
                "interpolation": self.interpolation,
                "antialias": self.antialias,
                "bounding_box_format": self.bounding_box_format,
            }
        )
        return config

    @classproperty
    def presets(cls):
        return builtin_presets(cls)

    @classmethod
    def from_preset(cls, preset, **kwargs):
        loader = get_preset_loader(preset)
        backbone_cls = loader.check_backbone_class()
        if cls.backbone_cls != backbone_cls:
            cls = find_subclass(preset, cls, backbone_cls)
        return loader.load_image_converter(cls, **kwargs)

    def save_to_preset(self, preset_dir):
        saver = get_preset_saver(preset_dir)
        saver.save_image_converter(self)
