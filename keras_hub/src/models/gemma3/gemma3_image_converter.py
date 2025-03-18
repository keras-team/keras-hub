import tensorflow as tf

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter

# from keras_hub.src.models.gemma3.gemma3_backbone import Gemma3Backbone
from keras_hub.src.utils.tensor_utils import preprocessing_function


def pad_image_list_ragged(
    image_list,
    image_height,
    image_width,
    image_max_length=5,
    pad_value=0,
):
    """Attempts to pad without list comprehension (likely to fail)."""

    if isinstance(image_list, tf.RaggedTensor):
        ragged_images = image_list
    elif isinstance(image_list, tf.Tensor):
        ragged_images = tf.RaggedTensor.from_tensor(image_list)
    else:
        print(type(image_list))
        ragged_images = tf.ragged.constant(image_list)

    batch_size = ragged_images.nrows()
    num_images = ragged_images.row_lengths()
    padded_images_dense = ragged_images.to_tensor(
        shape=[batch_size, image_max_length, image_height, image_width, 3],
        default_value=pad_value,
    )
    padded_images_dense = tf.convert_to_tensor(padded_images_dense)

    return padded_images_dense, num_images


@keras_hub_export("keras_hub.layers.Gemma3ImageConverter")
class Gemma3ImageConverter(ImageConverter):
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
        bounding_box_format: A string specifying the format of the bounding
            boxes, one of `"xyxy"`, `"rel_xyxy"`, `"xywh"`, `"center_xywh"`,
            `"yxyx"`, `"rel_yxyx"`. Specifies the format of the bounding boxes
            which will be resized to `image_size` along with the image. To pass
            bounding boxed to this layer, pass a dict with keys `"images"` and
            `"bounding_boxes"` when calling the layer.
        data_format: String, either `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs. `"channels_last"`
            corresponds to inputs with shape `(batch, height, width, channels)`
            while `"channels_first"` corresponds to inputs with shape
            `(batch, channels, height, width)`. It defaults to the
            `image_data_format` value found in your Keras config file at
            `~/.keras/keras.json`. If you never set it, then it will be
            `"channels_last"`.
        image_max_length: int. The maximum number of images per sample (padded).
            Defaults to `None`.

    Examples:
    ```python
    # Resize raw images and scale them to [0, 1].
    converter = keras_hub.layers.ImageConverter(
        image_size=(128, 128),
        scale=1. / 255,
        image_max_length=5,
    )
    converter(np.random.randint(0, 256, size=(2, 512, 512, 3)))

    # Resize images to the specific size needed for a Gemma3 preset.
    converter = keras_hub.layers.ImageConverter.from_preset(
        "gemma3_224" #Todo: update preset name
    )
    converter(np.random.randint(0, 256, size=(2, 512, 512, 3)))
    ```
    """

    # backbone_cls = Gemma3Backbone

    def __init__(
        self,
        image_size=None,
        scale=None,
        offset=None,
        crop_to_aspect_ratio=True,
        pad_to_aspect_ratio=False,
        interpolation="bilinear",
        bounding_box_format="yxyx",
        data_format=None,
        image_max_length=None,
        **kwargs,
    ):
        super().__init__(
            image_size=image_size,
            scale=scale,
            offset=offset,
            crop_to_aspect_ratio=crop_to_aspect_ratio,
            pad_to_aspect_ratio=pad_to_aspect_ratio,
            interpolation=interpolation,
            bounding_box_format=bounding_box_format,
            data_format=data_format,
            **kwargs,
        )
        self.image_max_length = image_max_length
        if image_max_length is None:
            raise ValueError(
                "image_max_length must be specified when pad_image_list is "
                "set to True."
            )

    def call(self, inputs):
        if isinstance(inputs, dict):
            x = inputs["images"]
        else:
            x = inputs

        # If images is `None`, return nothing.
        if x is None:
            return inputs

        # TODO: Figure out unbatched inputs.
        # x = [
        #     np.expand_dims(tensor, axis=0)
        #     if len(ops.shape(tensor)) == 3
        #     else np.array(tensor)
        #     for tensor in x
        # ]
        first_element_shape = tf.shape(x[0])

        padded_images, num_valid_images = pad_image_list_ragged(
            image_list=x,
            image_height=first_element_shape[-3],
            image_width=first_element_shape[-2],
            image_max_length=self.image_max_length,
            pad_value=0,
        )
        images = self.preprocess_images(padded_images)

        if isinstance(inputs, dict):
            return {
                "images": images,
                "num_valid_images": num_valid_images,
            }
        else:
            return images, num_valid_images

    @preprocessing_function
    def preprocess_images(self, images):
        original_shape = tf.shape(images)
        height = self.image_size[0] if self.image_size else None
        width = self.image_size[1] if self.image_size else None
        if self.image_size is not None:
            if len(original_shape) == 5:
                images = tf.reshape(
                    images,
                    [
                        -1,
                        original_shape[-3],
                        original_shape[-2],
                        original_shape[-1],
                    ],
                )
                images = self.resizing(images)
                images = tf.reshape(
                    images,
                    [
                        original_shape[0],
                        self.image_max_length,
                        height,
                        width,
                        3,
                    ],
                )
            else:
                images = self.resizing(images)
        # Allow dictionary input for handling bounding boxes.
        x = images
        if self.scale is not None:
            x = x * self._expand_non_channel_dims(self.scale, x)
        if self.offset is not None:
            x = x + self._expand_non_channel_dims(self.offset, x)
        # Pad images after all other preprocessing
        return images

    def get_config(self):
        config = super().get_config()
        config["image_max_length"] = self.image_max_length
        return config
