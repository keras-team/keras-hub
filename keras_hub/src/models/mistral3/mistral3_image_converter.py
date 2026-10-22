import numpy as np
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.mistral3.mistral3_backbone import Mistral3Backbone
from keras_hub.src.models.mistral3.mistral3_vision_encoder import (
    MISTRAL3_DEFAULT_SPATIAL_MERGE_SIZE,
)
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import tf

# CLIP normalization stats, in [0, 1] pixel-value units.
_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def compute_resize_size(height, width, longest_edge, patch_size):
    """Computes the resize target for one image.

    Scales `(height, width)` down (preserving aspect ratio) so its longest
    edge is at most `longest_edge`, then rounds each dimension up to the
    nearest multiple of `patch_size`.

    Args:
        height: int. The image's original height.
        width: int. The image's original width.
        longest_edge: int. The maximum allowed size of the longer side.
        patch_size: int. The patch size each output dimension must be a
            multiple of.

    Returns:
        `(resized_height, resized_width)` as plain Python ints.
    """
    ratio = max(height / longest_edge, width / longest_edge)
    if ratio > 1:
        height = int(height / ratio)
        width = int(width / ratio)

    resized_height = ((height - 1) // patch_size + 1) * patch_size
    resized_width = ((width - 1) // patch_size + 1) * patch_size
    return resized_height, resized_width


def _compute_resize_size_tf(height, width, longest_edge, patch_size):
    """Computes `compute_resize_size` with TensorFlow ops.

    The inputs can be scalar tensors with unknown static values.
    The computation uses `float64` to match the Python version.

    Args:
        height: scalar int tensor. The original height.
        width: scalar int tensor. The original width.
        longest_edge: int. The maximum allowed size of the longer side.
        patch_size: int. The patch size each output dimension must be a
            multiple of.

    Returns:
        `(resized_height, resized_width)` as scalar `int32` tensors.
    """
    height_f = tf.cast(height, "float64")
    width_f = tf.cast(width, "float64")
    ratio = tf.maximum(height_f / longest_edge, width_f / longest_edge)
    needs_resize = ratio > 1.0
    height = tf.where(needs_resize, tf.cast(height_f / ratio, "int32"), height)
    width = tf.where(needs_resize, tf.cast(width_f / ratio, "int32"), width)
    resized_height = ((height - 1) // patch_size + 1) * patch_size
    resized_width = ((width - 1) // patch_size + 1) * patch_size
    return resized_height, resized_width


# Bits of the signed integer weights. Torchvision stores its `uint8` resize
# weights as `int16`.
_WEIGHT_BITS = 15
_MAX_WEIGHT_PRECISION = 22


def _cubic_weight(x):
    """Evaluates the bicubic kernel (`a=-0.5`) with NumPy."""
    x = np.abs(x)
    near = ((-0.5 + 2) * x - (-0.5 + 3)) * x * x + 1
    far = (((x - 5) * x + 8) * x - 4) * -0.5
    return np.where(x < 1, near, np.where(x < 2, far, 0.0))


def _quantized_weights(size_in, size_out):
    """Computes the integer resize weights of one axis.

    This mirrors the `uint8` antialiased bicubic resize of torchvision.
    Each weight is `round(weight * 2**precision)`. The function picks the
    largest precision for which the largest weight still fits in `int16`.

    Args:
        size_in: int. The input size of the axis.
        size_out: int. The output size of the axis.

    Returns:
        `(weights, precision)`. `weights` is a `float64` array of shape
        `(size_out, size_in)` that holds integer values.
    """
    scale = size_in / size_out
    filter_scale = max(scale, 1.0)
    support = 2.0 * filter_scale
    center = (np.arange(size_out) + 0.5) * scale
    start = np.maximum(np.floor(center - support + 0.5), 0)
    stop = np.minimum(np.floor(center + support + 0.5), size_in)
    index = np.arange(size_in)[None, :]
    inside = (index >= start[:, None]) & (index < stop[:, None])
    weights = _cubic_weight((index + 0.5 - center[:, None]) / filter_scale)
    weights = weights * inside
    weights = weights / weights.sum(axis=1, keepdims=True)

    max_weight = np.abs(weights).max()
    precision = 0
    while (
        precision < _MAX_WEIGHT_PRECISION
        and int(0.5 + 2.0 ** (precision + 1) * max_weight) < 2**_WEIGHT_BITS
    ):
        precision += 1
    scaled = weights * 2.0**precision
    half = np.where(scaled < 0, -0.5, 0.5)
    return np.trunc(scaled + half), precision


def _resize_axis(image, axis, size_out):
    """Resizes one axis of an integer-valued image with NumPy.

    Args:
        image: float64 array. The pixel values are integers in `[0, 255]`.
        axis: int. The axis to resize.
        size_out: int. The output size of the axis.

    Returns:
        A `float64` array with integer values in `[0, 255]`.
    """
    weights, precision = _quantized_weights(image.shape[axis], size_out)
    image = np.tensordot(weights, image, axes=(1, axis))
    image = np.moveaxis(image, 0, axis)
    image = np.floor((image + 2.0 ** (precision - 1)) / 2.0**precision)
    return np.clip(image, 0, 255)


def _cubic_weight_tf(x):
    """Evaluates the bicubic kernel (`a=-0.5`) with TensorFlow ops."""
    x = tf.abs(x)
    near = ((-0.5 + 2) * x - (-0.5 + 3)) * x * x + 1
    far = (((x - 5) * x + 8) * x - 4) * -0.5
    return tf.where(x < 1, near, tf.where(x < 2, far, tf.zeros_like(x)))


def _quantized_weights_tf(size_in, size_out):
    """Computes `_quantized_weights` with TensorFlow ops.

    Args:
        size_in: scalar int tensor. The input size of the axis.
        size_out: scalar int tensor. The output size of the axis.

    Returns:
        `(weights, precision)`. `weights` is a `float64` tensor of shape
        `(size_out, size_in)`. `precision` is a scalar `float64` tensor.
    """
    size_in = tf.cast(size_in, "float64")
    size_out = tf.cast(size_out, "float64")
    scale = size_in / size_out
    filter_scale = tf.maximum(scale, 1.0)
    support = 2.0 * filter_scale
    center = (tf.range(size_out, dtype="float64") + 0.5) * scale
    start = tf.maximum(tf.floor(center - support + 0.5), 0.0)
    stop = tf.minimum(tf.floor(center + support + 0.5), size_in)
    index = tf.range(size_in, dtype="float64")[None, :]
    inside = (index >= start[:, None]) & (index < stop[:, None])
    weights = _cubic_weight_tf((index + 0.5 - center[:, None]) / filter_scale)
    weights = weights * tf.cast(inside, "float64")
    weights = weights / tf.reduce_sum(weights, axis=1, keepdims=True)

    max_weight = tf.reduce_max(tf.abs(weights))
    # The fit test is monotonic, so the precision is the count of passes.
    steps = tf.range(_MAX_WEIGHT_PRECISION, dtype="float64")
    fits = tf.floor(0.5 + 2.0 ** (steps + 1.0) * max_weight) < 2.0**_WEIGHT_BITS
    precision = tf.reduce_sum(tf.cast(fits, "float64"))
    scaled = weights * 2.0**precision
    weights = tf.where(
        scaled < 0, tf.math.ceil(scaled - 0.5), tf.floor(scaled + 0.5)
    )
    return weights, precision


def _resize_axis_tf(images, axis, size_out):
    """Resizes one axis of integer-valued `(N, H, W, C)` images.

    Args:
        images: float64 tensor. The pixel values are integers in `[0, 255]`.
        axis: int. The axis to resize, `1` for height or `2` for width.
        size_out: scalar int tensor. The output size of the axis.

    Returns:
        A `float64` tensor with integer values in `[0, 255]`.
    """
    weights, precision = _quantized_weights_tf(tf.shape(images)[axis], size_out)
    if axis == 1:
        images = tf.einsum("oh,nhwc->nowc", weights, images)
    else:
        images = tf.einsum("ow,nhwc->nhoc", weights, images)
    images = tf.floor((images + 2.0 ** (precision - 1.0)) / 2.0**precision)
    return tf.clip_by_value(images, 0.0, 255.0)


@keras_hub_export("keras_hub.layers.Mistral3ImageConverter")
class Mistral3ImageConverter(ImageConverter):
    """Converts raw images into `Mistral3Backbone`'s multimodal inputs.

    Each image is resized so its longest edge is at most `longest_edge`
    pixels (aspect ratio preserved), then rounded up to a `patch_size`
    multiple, matching HF's `PixtralImageProcessor`. Since every image in a
    call can resize to a different shape, resizing runs in a Python loop
    over `inputs` (a list of variable-size images) rather than through the
    base class's single-`Resizing`-layer `call()`.

    The TensorFlow path (`tf.data`) resizes every image in a call to one
    shape. A dense batch tensor has one image size, so this path needs
    batches of same-size images. The Python path resizes each image on its
    own and pads the batch to the largest image.

    Args:
        longest_edge: int. The maximum size of an image's longer side after
            resizing. Defaults to `1540`.
        patch_size: int. The vision encoder's patch size. Defaults to `14`.
        spatial_merge_size: int. The number of patches merged together per
            side by the multimodal projector's patch merger. Resized image
            dimensions are rounded up to a multiple of
            `patch_size * spatial_merge_size`. Defaults to `2`.
        image_mean: list of 3 floats or `None`. The per-channel mean, in
            `[0, 1]` pixel units. Defaults to the CLIP mean.
        image_std: list of 3 floats or `None`. The per-channel standard
            deviation, in `[0, 1]` pixel units. Defaults to the CLIP
            standard deviation.
        rescale_factor: float or `None`. The factor that maps `[0, 255]`
            pixels to `[0, 1]`. Defaults to `1 / 255`.

    The converter normalizes with `(x - mean / rescale_factor) /
    (std / rescale_factor)` in `float32`. HF's `PixtralImageProcessor` uses
    the same arithmetic, so the output pixels are identical.
    """

    backbone_cls = Mistral3Backbone

    def __init__(
        self,
        longest_edge=1540,
        patch_size=14,
        spatial_merge_size=MISTRAL3_DEFAULT_SPATIAL_MERGE_SIZE,
        image_mean=None,
        image_std=None,
        rescale_factor=None,
        **kwargs,
    ):
        if image_mean is None:
            if "scale" in kwargs or "offset" in kwargs:
                raise ValueError(
                    "`scale` and `offset` are not supported. Pass "
                    "`image_mean`, `image_std` and `rescale_factor` instead."
                )
            image_mean = _CLIP_MEAN
        if image_std is None:
            image_std = _CLIP_STD
        if rescale_factor is None:
            rescale_factor = 1.0 / 255.0
        # The base class gets the equivalent `scale` and `offset`. The
        # converter itself normalizes with `image_mean` and `image_std`.
        # `get_config` of the base class returns `scale` and `offset`, so
        # drop them from a deserialized config.
        kwargs.pop("scale", None)
        kwargs.pop("offset", None)
        # `image_size=None` skips the base class's `Resizing` sublayer,
        # since Pixtral's resize target is dynamic per image. `dtype` is
        # always float32, independent of the model's compute dtype. Both
        # are hardcoded below, so drop any incoming values (e.g. from a
        # deserialized config) rather than conflict with them.
        kwargs.pop("dtype", None)
        kwargs.pop("image_size", None)
        # The call path pads and resizes with an antialiased bicubic kernel.
        # Pass these values so the saved config describes that behavior.
        kwargs.pop("crop_to_aspect_ratio", None)
        kwargs.pop("interpolation", None)
        kwargs.pop("antialias", None)
        super().__init__(
            image_size=None,
            scale=[rescale_factor / s for s in image_std],
            offset=[-m / s for m, s in zip(image_mean, image_std)],
            crop_to_aspect_ratio=False,
            interpolation="bicubic",
            antialias=True,
            dtype="float32",
            **kwargs,
        )
        self.longest_edge = longest_edge
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.image_mean = list(image_mean)
        self.image_std = list(image_std)
        self.rescale_factor = rescale_factor
        # HF multiplies the `float32` mean and std by `1 / rescale_factor`
        # in `float32`, so the same rounding steps are needed here.
        to_pixels = np.float32(1.0 / rescale_factor)
        self._mean_pixels = np.array(image_mean, dtype="float32") * to_pixels
        self._std_pixels = np.array(image_std, dtype="float32") * to_pixels

    def _call_python(self, inputs):
        unbatched = (
            not isinstance(inputs, (list, tuple))
            and len(ops.shape(inputs)) == 3
        )
        if unbatched:
            inputs = [inputs]

        # HF's `PixtralProcessor` rounds resized dimensions to a multiple of
        # `patch_size * spatial_merge_size`, not `patch_size` alone, so that
        # the patch grid divides evenly into the merged patch-merger grid.
        merge_patch_size = self.patch_size * self.spatial_merge_size
        resized_images = []
        image_sizes = []
        for image in inputs:
            image = ops.convert_to_numpy(image).astype("float32")
            height, width = image.shape[0], image.shape[1]
            resized_height, resized_width = compute_resize_size(
                height, width, self.longest_edge, merge_patch_size
            )
            # HF resizes the `uint8` image with torchvision. Torchvision
            # resizes the width first, then the height, with integer
            # weights. The same integer passes reproduce its output.
            image = np.clip(np.round(image), 0, 255).astype("float64")
            image = _resize_axis(image, 1, resized_width)
            image = _resize_axis(image, 0, resized_height)
            image = image.astype("float32")
            image = (image - self._mean_pixels) / self._std_pixels
            # Channels-last `(H, W, 3)` -> channels-first `(3, H, W)`, to
            # match `Mistral3Backbone`'s `pixel_values` input layout.
            image = np.transpose(image, (2, 0, 1))
            resized_images.append(image)
            image_sizes.append((resized_height, resized_width))

        max_height = max(size[0] for size in image_sizes)
        max_width = max(size[1] for size in image_sizes)

        padded_images = []
        for image, (resized_height, resized_width) in zip(
            resized_images, image_sizes
        ):
            pad_height = max_height - resized_height
            pad_width = max_width - resized_width
            padded_images.append(
                np.pad(
                    image,
                    ((0, 0), (0, pad_height), (0, pad_width)),
                )
            )

        pixel_values = np.stack(padded_images, axis=0).astype("float32")
        image_sizes = np.array(image_sizes, dtype="int32")
        if unbatched:
            return pixel_values[0], image_sizes[0]
        return pixel_values, image_sizes

    @preprocessing_function
    def _call_tf(self, inputs):
        images = tf.cast(inputs, "float32")
        unbatched = len(images.shape) == 3
        if unbatched:
            images = tf.expand_dims(images, axis=0)

        merge_patch_size = self.patch_size * self.spatial_merge_size
        # Use the dynamic shape. The static height and width are `None`
        # under `tf.data` when the images have unknown size.
        image_shape = tf.shape(images)
        resized_height, resized_width = _compute_resize_size_tf(
            image_shape[1], image_shape[2], self.longest_edge, merge_patch_size
        )
        # Same integer passes as `_call_python`: width, then height.
        images = tf.cast(images, "float64")
        images = tf.clip_by_value(tf.round(images), 0.0, 255.0)
        images = _resize_axis_tf(images, 2, resized_width)
        images = _resize_axis_tf(images, 1, resized_height)
        images = tf.cast(images, "float32")
        mean = tf.constant(self._mean_pixels, dtype="float32")
        std = tf.constant(self._std_pixels, dtype="float32")
        images = (images - mean) / std
        images = tf.transpose(images, (0, 3, 1, 2))
        num_images = tf.shape(images)[0]
        image_sizes = tf.tile(
            tf.stack([resized_height, resized_width])[tf.newaxis, :],
            (num_images, 1),
        )
        if unbatched:
            return images[0], image_sizes[0]
        return images, image_sizes

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "longest_edge": self.longest_edge,
                "patch_size": self.patch_size,
                "spatial_merge_size": self.spatial_merge_size,
                "image_mean": self.image_mean,
                "image_std": self.image_std,
                "rescale_factor": self.rescale_factor,
            }
        )
        return config
