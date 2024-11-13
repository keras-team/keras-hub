import math

import keras
import numpy as np
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.preprocessing_layer import (
    PreprocessingLayer,
)
from keras_hub.src.utils.keras_utils import standardize_data_format
from keras_hub.src.utils.preset_utils import builtin_presets
from keras_hub.src.utils.preset_utils import find_subclass
from keras_hub.src.utils.preset_utils import get_preset_loader
from keras_hub.src.utils.preset_utils import get_preset_saver
from keras_hub.src.utils.python_utils import classproperty
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export("keras_hub.layers.ImageConverter")
class ImageConverter(PreprocessingLayer):
    """Preprocess raw images into model ready inputs.

    This class converts from raw images to model ready inputs. This conversion
    proceeds in the following steps:

    1. Resize the image using to `image_size`. If `image_size` is `None`, this
       step will be skipped.
    2. Rescale the image by multiplying by `scale`, which can be either global
       or per channel. If `scale` is `None`, this step will be skipped.
    3. Offset the image by adding `offset`, which can be either global
       or per channel. If `offset` is `None`, this step will be skipped.

    The layer will take as input a raw image tensor in the channels last or
    channels first format, and output a preprocessed image input for modeling.
    This tensor can be batched (rank 4), or unbatched (rank 3).

    This layer can be used with the `from_preset()` constructor to load a layer
    that will rescale and resize an image for a specific pretrained model.
    Using the layer this way allows writing preprocessing code that does not
    need updating when switching between model checkpoints.

    Args:
        image_size: `(int, int)` tuple or `None`. The output size of the image,
            not including the channels axis. If `None`, the input will not be
            resized.
        scale: float, tuple of floats, or `None`. The scale to apply to the
            inputs. If `scale` is a single float, the entire input will be
            multiplied by `scale`. If `scale` is a tuple, it's assumed to
            contain per-channel scale value multiplied against each channel of
            the input images. If `scale` is `None`, no scaling is applied.
        offset: float, tuple of floats, or `None`. The offset to apply to the
            inputs. If `offset` is a single float, the entire input will be
            summed with `offset`. If `offset` is a tuple, it's assumed to
            contain per-channel offset value summed against each channel of the
            input images. If `offset` is `None`, no scaling is applied.
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
    # Resize raw images and scale them to [0, 1].
    converter = keras_hub.layers.ImageConverter(
        image_size=(128, 128),
        scale=1. / 255,
    )
    converter(np.random.randint(0, 256, size=(2, 512, 512, 3)))

    # Resize images to the specific size needed for a PaliGemma preset.
    converter = keras_hub.layers.ImageConverter.from_preset(
        "pali_gemma_3b_224"
    )
    converter(np.random.randint(0, 256, size=(2, 512, 512, 3)))
    ```
    """

    backbone_cls = None

    def __init__(
        self,
        image_size=None,
        scale=None,
        offset=None,
        crop_to_aspect_ratio=True,
        interpolation="bilinear",
        data_format=None,
        **kwargs,
    ):
        # TODO: old arg names. Delete this block after resaving Kaggle assets.
        if "height" in kwargs and "width" in kwargs:
            image_size = (kwargs.pop("height"), kwargs.pop("width"))
        if "variance" in kwargs and "mean" in kwargs:
            std = [math.sqrt(v) for v in kwargs.pop("variance")]
            scale = [scale / s for s in std]
            offset = [-m / s for m, s in zip(kwargs.pop("mean"), std)]

        super().__init__(**kwargs)

        # Create the `Resizing` layer here even if it's not being used. That
        # allows us to make `image_size` a settable property.
        self.resizing = keras.layers.Resizing(
            height=image_size[0] if image_size else None,
            width=image_size[1] if image_size else None,
            crop_to_aspect_ratio=crop_to_aspect_ratio,
            interpolation=interpolation,
            data_format=data_format,
            dtype=self.dtype_policy,
            name="resizing",
        )
        self.scale = scale
        self.offset = offset
        self.crop_to_aspect_ratio = crop_to_aspect_ratio
        self.interpolation = interpolation
        self.data_format = standardize_data_format(data_format)

    @property
    def image_size(self):
        """Settable tuple of `(height, width)` ints. The output image shape."""
        if self.resizing.height is None:
            return None
        return (self.resizing.height, self.resizing.width)

    @image_size.setter
    def image_size(self, value):
        if value is None:
            value = (None, None)
        self.resizing.height = value[0]
        self.resizing.width = value[1]

    @preprocessing_function
    def call(self, inputs):
        x = inputs
        if self.image_size is not None:
            x = self.resizing(x)
        if self.scale is not None:
            x = x * self._expand_non_channel_dims(self.scale, x)
        if self.offset is not None:
            x = x + self._expand_non_channel_dims(self.offset, x)
        return x

    def _expand_non_channel_dims(self, value, inputs):
        unbatched = len(ops.shape(inputs)) == 3
        channels_first = self.data_format == "channels_first"
        if unbatched:
            broadcast_dims = (1, 2) if channels_first else (0, 1)
        else:
            broadcast_dims = (0, 2, 3) if channels_first else (0, 1, 2)
        # If inputs are not a tensor type, return a numpy array.
        # This might happen when running under tf.data.
        if ops.is_tensor(inputs):
            # preprocessing decorator moves tensors to cpu in torch backend and
            # processed on CPU, and then converted back to the appropriate
            # device (potentially GPU) after preprocessing.
            if keras.backend.backend() == "torch" and self.image_size is None:
                return ops.expand_dims(value, broadcast_dims).cpu()
            return ops.expand_dims(value, broadcast_dims)
        else:
            return np.expand_dims(value, broadcast_dims)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_size": self.image_size,
                "scale": self.scale,
                "offset": self.offset,
                "interpolation": self.interpolation,
                "crop_to_aspect_ratio": self.crop_to_aspect_ratio,
            }
        )
        return config

    @classproperty
    def presets(cls):
        """List built-in presets for an `ImageConverter` subclass."""
        return builtin_presets(cls)

    @classmethod
    def from_preset(
        cls,
        preset,
        **kwargs,
    ):
        """Instantiate a `keras_hub.layers.ImageConverter` from a model preset.

        A preset is a directory of configs, weights and other file assets used
        to save and load a pre-trained model. The `preset` can be passed as
        one of:

        1. a built-in preset identifier like `'pali_gemma_3b_224'`
        2. a Kaggle Models handle like
           `'kaggle://user/paligemma/keras/pali_gemma_3b_224'`
        3. a Hugging Face handle like `'hf://user/pali_gemma_3b_224'`
        4. a path to a local preset directory like `'./pali_gemma_3b_224'`

        You can run `cls.presets.keys()` to list all built-in presets available
        on the class.

        Args:
            preset: string. A built-in preset identifier, a Kaggle Models
                handle, a Hugging Face handle, or a path to a local directory.
            load_weights: bool. If `True`, the weights will be loaded into the
                model architecture. If `False`, the weights will be randomly
                initialized.

        Examples:
        ```python
        batch = np.random.randint(0, 256, size=(2, 512, 512, 3))

        # Resize images for `"pali_gemma_3b_224"`.
        converter = keras_hub.layers.ImageConverter.from_preset(
            "pali_gemma_3b_224"
        )
        converter(batch) # Output shape: (2, 224, 224, 3)

        # Resize images for `"pali_gemma_3b_448"` without cropping.
        converter = keras_hub.layers.ImageConverter.from_preset(
            "pali_gemma_3b_448",
            crop_to_aspect_ratio=False,
        )
        converter(batch) # Output shape: (2, 448, 448, 3)
        ```
        """
        loader = get_preset_loader(preset)
        backbone_cls = loader.check_backbone_class()
        if cls.backbone_cls != backbone_cls:
            cls = find_subclass(preset, cls, backbone_cls)
        return loader.load_image_converter(cls, **kwargs)

    def save_to_preset(self, preset_dir):
        """Save image converter to a preset directory.

        Args:
            preset_dir: The path to the local model preset directory.
        """
        saver = get_preset_saver(preset_dir)
        saver.save_image_converter(self)
