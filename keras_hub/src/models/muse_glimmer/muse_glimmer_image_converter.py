import itertools

import keras
import numpy as np
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.muse_glimmer.muse_glimmer_backbone import (
    MuseGlimmerBackbone,
)
from keras_hub.src.utils.tensor_utils import in_tf_function
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import tf


def _smart_resize(height, width, patch_size, merge_size, max_tokens):
    """Select a patch grid that preserves the input aspect ratio.

    `height` and `width` are Python ints. If two grids have the same aspect
    error, the first grid in `list(set(...))` order wins.
    """
    resize_patch_size = patch_size * merge_size
    # Use Python floats (float64), like the HF `smart_resize`. A float32
    # grid picks a different patch grid for a few sizes.
    ideal_height = height / resize_patch_size
    ideal_width = width / resize_patch_size
    ratio = ideal_width / ideal_height
    if ideal_height * ideal_width > max_tokens:
        ideal_height = (max_tokens / ratio) ** 0.5
        ideal_width = ideal_height * ratio

    candidates = list(
        set(
            itertools.product(
                (int(np.floor(ideal_height)), int(np.ceil(ideal_height))),
                (int(np.floor(ideal_width)), int(np.ceil(ideal_width))),
            )
        )
    )
    candidates = [
        (grid_h, grid_w)
        for grid_h, grid_w in candidates
        if grid_h >= 1 and grid_w >= 1 and grid_h * grid_w <= max_tokens
    ]
    if not candidates:
        candidates = [(max(round(ideal_height), 1), max(round(ideal_width), 1))]
    selected = min(
        candidates, key=lambda grid: abs(grid[0] / grid[1] - height / width)
    )
    return selected[0] * resize_patch_size, selected[1] * resize_patch_size


def _smart_resize_tf(height, width, patch_size, merge_size, max_tokens):
    """Select a patch grid inside a TensorFlow graph.

    Use this function only when the image size is unknown until run
    time. For a static size, `_smart_resize` gives Python ints, so the
    converter outputs have static shapes.

    The function calls `_smart_resize` through `tf.numpy_function`. The
    tie-break of `_smart_resize` depends on Python `set` order, and a
    graph cannot reproduce that order. A saved graph cannot hold this
    op, and XLA cannot compile it.
    """

    def select_grid(height, width):
        target = _smart_resize(
            int(height), int(width), patch_size, merge_size, max_tokens
        )
        return np.int32(target[0]), np.int32(target[1])

    target_h, target_w = tf.numpy_function(
        select_grid,
        [height, width],
        [tf.int32, tf.int32],
        stateful=False,
    )
    target_h.set_shape([])
    target_w.set_shape([])
    return target_h, target_w


def _lanczos3_kernel(x):
    x = np.asarray(x, dtype="float64")
    return np.where(np.abs(x) < 3.0, np.sinc(x) * np.sinc(x / 3.0), 0.0)


def _fixed_point_lanczos_weights(in_size, out_size):
    """Return the int32 weight matrix and the precision for one resize pass.

    The weights follow the torchvision CPU resize of uint8 images. The
    precision is the largest number of bits that keeps every scaled weight
    below `2**15`. Each weight rounds half away from zero. Float weights
    give a different uint8 value for about 1% of the values.
    """
    scale = in_size / out_size
    support = 3.0 * max(scale, 1.0)
    inv_scale = 1.0 / max(scale, 1.0)
    rows = []
    for index in range(out_size):
        center = scale * (index + 0.5)
        start = max(int(center - support + 0.5), 0)
        stop = min(int(center + support + 0.5), in_size)
        weights = _lanczos3_kernel(
            (np.arange(start, stop) - center + 0.5) * inv_scale
        )
        rows.append((start, weights / weights.sum()))
    max_weight = max(np.abs(weights).max() for _, weights in rows)
    precision = 0
    while (
        precision < 22 and int(0.5 + max_weight * 2 ** (precision + 1)) < 2**15
    ):
        precision += 1
    matrix = np.zeros((out_size, in_size), dtype="int32")
    for index, (start, weights) in enumerate(rows):
        scaled = weights * 2.0**precision
        matrix[index, start : start + len(weights)] = np.where(
            weights < 0, scaled - 0.5, scaled + 0.5
        ).astype("int64")
    return matrix, precision


def _resize_uint8_lanczos(images, target_h, target_w):
    """Resize integer `(..., height, width, 3)` images with Lanczos.

    The result matches the torchvision CPU resize of uint8 images. Each
    pass uses integer weights, adds half a unit, shifts right, and clips
    to `[0, 255]`. The width pass runs first.
    """
    rank = len(ops.shape(images))
    if rank == 3:
        images = ops.expand_dims(images, 0)
    images = ops.cast(images, "int32")
    _, in_h, in_w, _ = ops.shape(images)
    if in_w != target_w:
        matrix, precision = _fixed_point_lanczos_weights(in_w, target_w)
        images = ops.einsum(
            "ow,nhwc->nhoc", ops.convert_to_tensor(matrix), images
        )
        images = ops.floor_divide(
            images + (1 << (precision - 1)), 1 << precision
        )
        images = ops.clip(images, 0, 255)
    if in_h != target_h:
        matrix, precision = _fixed_point_lanczos_weights(in_h, target_h)
        images = ops.einsum(
            "oh,nhwc->nowc", ops.convert_to_tensor(matrix), images
        )
        images = ops.floor_divide(
            images + (1 << (precision - 1)), 1 << precision
        )
        images = ops.clip(images, 0, 255)
    return images[0] if rank == 3 else images


def _fixed_point_lanczos_weights_tf(in_size, out_size):
    """Return `_fixed_point_lanczos_weights` for ints or scalar tensors."""
    if isinstance(in_size, int) and isinstance(out_size, int):
        matrix, precision = _fixed_point_lanczos_weights(in_size, out_size)
        return tf.constant(matrix), tf.constant(precision, dtype="int32")

    def compute(in_size, out_size):
        matrix, precision = _fixed_point_lanczos_weights(
            int(in_size), int(out_size)
        )
        return matrix, np.int32(precision)

    matrix, precision = tf.numpy_function(
        compute, [in_size, out_size], [tf.int32, tf.int32], stateful=False
    )
    matrix.set_shape([None, None])
    precision.set_shape([])
    return matrix, precision


def _resize_uint8_lanczos_tf(images, target_h, target_w):
    """Run `_resize_uint8_lanczos` on a `(batch, h, w, 3)` TF tensor."""
    images = tf.cast(images, "int32")
    in_h = images.shape[1] or tf.shape(images)[1]
    in_w = images.shape[2] or tf.shape(images)[2]
    for equation, in_size, out_size in (
        ("ow,nhwc->nhoc", in_w, target_w),
        ("oh,nhwc->nowc", in_h, target_h),
    ):
        matrix, precision = _fixed_point_lanczos_weights_tf(in_size, out_size)
        images = tf.einsum(equation, matrix, images)
        half = tf.bitwise.left_shift(1, precision - 1)
        images = tf.bitwise.right_shift(images + half, precision)
        images = tf.clip_by_value(images, 0, 255)
    return images


def _resize(images, orig_h, orig_w, target_h, target_w, method, antialias):
    """Resize a `(..., height, width, 3)` image or video with Keras ops."""
    rank = len(ops.shape(images))
    # The PyTorch backend does not support Lanczos interpolation in
    # `ops.image.resize`. Fall back to `scale_and_translate`, the
    # primitive `resize` itself uses on other backends for this case.
    if keras.backend.backend() == "torch" and method in (
        "lanczos3",
        "lanczos5",
    ):
        leading_dims = tuple(int(d) for d in ops.shape(images)[:-3])
        scale = ops.array(
            [target_h / orig_h, target_w / orig_w], dtype="float32"
        )
        return ops.image.scale_and_translate(
            images,
            leading_dims + (target_h, target_w, 3),
            scale=scale,
            translation=ops.zeros((2,), dtype="float32"),
            spatial_dims=(rank - 3, rank - 2),
            method=method,
            antialias=antialias,
        )
    if rank == 3:
        images = ops.expand_dims(images, 0)
    images = ops.image.resize(
        images,
        size=(target_h, target_w),
        interpolation=method,
        antialias=antialias,
        data_format="channels_last",
    )
    return images[0] if rank == 3 else images


def _resize_pixels(
    images, orig_h, orig_w, target_h, target_w, method, antialias
):
    """Resize images with Keras ops to float32 values in `[0, 255]`."""
    is_integer = keras.backend.is_int_dtype(images.dtype)
    if is_integer and method == "lanczos3" and antialias:
        images = _resize_uint8_lanczos(images, target_h, target_w)
    elif is_integer:
        # Matches torchvision's separable uint8 resize: width pass,
        # round to uint8 range, then height pass, round again.
        images = _resize(
            images, orig_h, orig_w, orig_h, target_w, method, antialias
        )
        # The TF backend's `ops.image.resize` preserves integer
        # dtypes, so clip/round need a float cast first.
        images = ops.cast(images, "float32")
        images = ops.round(ops.clip(images, 0.0, 255.0))
        images = _resize(
            images, orig_h, target_w, target_h, target_w, method, antialias
        )
        images = ops.round(ops.clip(images, 0.0, 255.0))
    else:
        images = _resize(
            images, orig_h, orig_w, target_h, target_w, method, antialias
        )
    images = ops.cast(images, "float32")
    return ops.clip(images, 0.0, 255.0)


def _resize_pixels_tf(
    images, input_is_integer, orig_h, target_h, target_w, method, antialias
):
    """Resize float32 images in a TF graph to values in `[0, 255]`."""
    unbatched = images.shape.rank == 3
    if unbatched:
        images = images[tf.newaxis]
    if input_is_integer and method == "lanczos3" and antialias:
        images = _resize_uint8_lanczos_tf(images, target_h, target_w)
        images = tf.cast(images, "float32")
    elif input_is_integer:
        # Matches torchvision's separable uint8 resize: width pass,
        # round to uint8 range, then height pass, round again.
        images = tf.image.resize(
            images, (orig_h, target_w), method=method, antialias=antialias
        )
        images = tf.round(tf.clip_by_value(images, 0.0, 255.0))
        images = tf.image.resize(
            images, (target_h, target_w), method=method, antialias=antialias
        )
        images = tf.round(tf.clip_by_value(images, 0.0, 255.0))
    else:
        images = tf.image.resize(
            images, (target_h, target_w), method=method, antialias=antialias
        )
    if unbatched:
        images = images[0]
    return tf.clip_by_value(images, 0.0, 255.0)


def _normalize(images, image_converter, compute_dtype, image_mean, image_std):
    """Normalize channels-last `images` like the HF processor.

    The function computes `(images - image_mean) / image_std` in float32.
    This gives the same float32 values as HF. If `image_mean` is `None`,
    the function returns `images` unchanged.
    """
    if image_mean is None:
        return images
    images, mean = image_converter._convert_types(
        images, np.asarray(image_mean, dtype="float32"), "float32"
    )
    _, std = image_converter._convert_types(
        images, np.asarray(image_std, dtype="float32"), "float32"
    )
    images = (images - mean) / std
    images, _ = image_converter._convert_types(images, mean, compute_dtype)
    return images


@keras_hub_export("keras_hub.layers.MuseGlimmerImageConverter")
class MuseGlimmerImageConverter(ImageConverter):
    """Image preprocessor for MuseGlimmer.

    The converter takes one unbatched image. The image has the shape
    `(height, width, 3)` for `data_format="channels_last"` and
    `(3, height, width)` for `data_format="channels_first"`.
    The converter resizes an image to a patch-grid-aligned size. It
    extracts `patch_size x patch_size` patches. It duplicates each patch
    along the temporal axis (`patch_temporal`). It flattens each patch to
    the vector that `MuseGlimmerVisionPatchEmbedder` expects.

    Args:
        patch_size: int. Spatial patch size in pixels. Defaults to `14`.
        patch_temporal: int. Temporal patch size. Defaults to `2`.
        merge_size: int. Spatial merge (pixel-shuffle) factor. Defaults to
            `2`.
        max_image_tokens: int. Maximum number of merged vision tokens per
            image. Defaults to `4096`.
        image_mean: float, list of floats, or `None`. Per-channel mean in
            pixel units `[0, 255]`. The converter computes
            `(image - image_mean) / image_std` in float32. Defaults to `None`.
        image_std: float, list of floats, or `None`. Per-channel standard
            deviation in pixel units `[0, 255]`. Set it together with
            `image_mean`. Defaults to `None`.
        interpolation: string. Resize interpolation method. Defaults to
            `"lanczos3"`.
        antialias: bool. If `True`, the resize filters the image to avoid
            aliasing. Defaults to `True`.
    """

    backbone_cls = MuseGlimmerBackbone

    def __init__(
        self,
        patch_size=14,
        patch_temporal=2,
        merge_size=2,
        max_image_tokens=4096,
        image_mean=None,
        image_std=None,
        interpolation="lanczos3",
        antialias=True,
        **kwargs,
    ):
        super().__init__(
            interpolation=interpolation, antialias=antialias, **kwargs
        )
        for name in ("scale", "offset"):
            if kwargs.get(name) is not None:
                raise ValueError(
                    f"`{name}` is not supported. Use `image_mean` and "
                    "`image_std`."
                )
        if (image_mean is None) != (image_std is None):
            raise ValueError(
                "Set `image_mean` and `image_std` together. Received "
                f"`image_mean={image_mean}` and `image_std={image_std}`."
            )
        self.patch_size = patch_size
        self.patch_temporal = patch_temporal
        self.merge_size = merge_size
        self.max_image_tokens = max_image_tokens
        self.image_mean = image_mean
        self.image_std = image_std

    @preprocessing_function
    def call(self, inputs):
        if in_tf_function():
            return self._call_tf(inputs)
        return self._call_ops(inputs)

    def _check_rank(self, rank):
        if rank != 3:
            raise ValueError(
                "`MuseGlimmerImageConverter` expects one unbatched image "
                "with rank 3. Call the converter once for each image. "
                f"Received an input with rank {rank}."
            )

    def _call_tf(self, inputs):
        self._check_rank(inputs.shape.rank)
        if self.data_format == "channels_first":
            inputs = tf.transpose(inputs, (1, 2, 0))
        input_is_integer = tf.as_dtype(inputs.dtype).is_integer
        image = tf.cast(inputs, "float32")
        orig_h, orig_w = image.shape[0], image.shape[1]
        smart_resize = _smart_resize
        if orig_h is None or orig_w is None:
            orig_h, orig_w = tf.shape(image)[0], tf.shape(image)[1]
            smart_resize = _smart_resize_tf
        target_h, target_w = smart_resize(
            orig_h,
            orig_w,
            self.patch_size,
            self.merge_size,
            self.max_image_tokens,
        )
        image = _resize_pixels_tf(
            image,
            input_is_integer,
            orig_h,
            target_h,
            target_w,
            self.interpolation,
            self.antialias,
        )
        image = _normalize(
            image,
            self,
            self.compute_dtype,
            self.image_mean,
            self.image_std,
        )

        grid_h, grid_w = (
            target_h // self.patch_size,
            target_w // self.patch_size,
        )
        image = tf.reshape(
            image, (grid_h, self.patch_size, grid_w, self.patch_size, 3)
        )
        image = tf.transpose(image, (0, 2, 4, 1, 3))
        num_patches = grid_h * grid_w
        image = tf.reshape(
            image, (num_patches, self.patch_size * self.patch_size * 3)
        )
        image = tf.tile(image[:, tf.newaxis, :], [1, self.patch_temporal, 1])
        image = tf.reshape(
            image,
            (
                num_patches,
                self.patch_temporal * self.patch_size * self.patch_size * 3,
            ),
        )
        grid_thw = tf.stack([tf.constant(1, dtype="int32"), grid_h, grid_w])
        return {"patches": image, "grid_thw": grid_thw}

    def _call_ops(self, inputs):
        self._check_rank(len(ops.shape(inputs)))
        image = inputs
        if self.data_format == "channels_first":
            image = ops.transpose(image, (1, 2, 0))
        orig_h, orig_w = int(ops.shape(image)[0]), int(ops.shape(image)[1])
        target_h, target_w = _smart_resize(
            orig_h,
            orig_w,
            self.patch_size,
            self.merge_size,
            self.max_image_tokens,
        )
        image = _resize_pixels(
            image,
            orig_h,
            orig_w,
            target_h,
            target_w,
            self.interpolation,
            self.antialias,
        )
        image = _normalize(
            image,
            self,
            self.compute_dtype,
            self.image_mean,
            self.image_std,
        )

        grid_h, grid_w = (
            target_h // self.patch_size,
            target_w // self.patch_size,
        )
        image = ops.reshape(
            image, (grid_h, self.patch_size, grid_w, self.patch_size, 3)
        )
        image = ops.transpose(image, (0, 2, 4, 1, 3))
        num_patches = grid_h * grid_w
        image = ops.reshape(
            image, (num_patches, self.patch_size * self.patch_size * 3)
        )
        image = ops.tile(ops.expand_dims(image, 1), (1, self.patch_temporal, 1))
        image = ops.reshape(
            image,
            (
                num_patches,
                self.patch_temporal * self.patch_size * self.patch_size * 3,
            ),
        )
        grid_thw = ops.stack([ops.array(1, dtype="int32"), grid_h, grid_w])
        return {"patches": image, "grid_thw": grid_thw}

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "patch_temporal": self.patch_temporal,
                "merge_size": self.merge_size,
                "max_image_tokens": self.max_image_tokens,
                "image_mean": self.image_mean,
                "image_std": self.image_std,
            }
        )
        # The converter ignores these base class settings.
        config.pop("scale", None)
        config.pop("offset", None)
        config.pop("crop_to_aspect_ratio", None)
        return config
