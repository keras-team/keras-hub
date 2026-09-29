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

    `height` and `width` are Python ints. The math uses float32 NumPy.
    """
    resize_patch_size = patch_size * merge_size
    ideal_grid = np.array(
        [height / resize_patch_size, width / resize_patch_size],
        dtype="float32",
    )
    ratio = ideal_grid[1] / ideal_grid[0]
    limited_height = np.sqrt(np.float32(max_tokens) / ratio)
    if ideal_grid[0] * ideal_grid[1] > max_tokens:
        ideal_grid = np.stack([limited_height, limited_height * ratio])

    lower_grid = np.floor(ideal_grid)
    upper_grid = np.ceil(ideal_grid)
    candidates = np.array(
        [
            [lower_grid[0], lower_grid[1]],
            [lower_grid[0], upper_grid[1]],
            [upper_grid[0], lower_grid[1]],
            [upper_grid[0], upper_grid[1]],
        ],
        dtype="float32",
    )
    valid = (
        (candidates[:, 0] >= 1)
        & (candidates[:, 1] >= 1)
        & (candidates[:, 0] * candidates[:, 1] <= max_tokens)
    )
    if valid.any():
        aspect_error = np.abs(
            candidates[:, 0] / candidates[:, 1] - np.float32(height / width)
        )
        aspect_error = np.where(valid, aspect_error, np.float32(1e9))
        selected = candidates[np.argmin(aspect_error)]
    else:
        selected = np.maximum(np.round(ideal_grid), 1)
    return (
        int(selected[0]) * resize_patch_size,
        int(selected[1]) * resize_patch_size,
    )


def _smart_resize_tf(height, width, patch_size, merge_size, max_tokens):
    """Select a patch grid inside a TensorFlow graph.

    Use this function only when the image size is unknown until run
    time. For a static size, `_smart_resize` gives Python ints, so the
    converter outputs have static shapes.
    """
    resize_patch_size = tf.cast(patch_size * merge_size, "float32")
    ideal_grid = tf.cast(tf.stack([height, width]), "float32")
    ideal_grid = ideal_grid / resize_patch_size
    ratio = ideal_grid[1] / ideal_grid[0]
    limited_height = tf.sqrt(tf.cast(max_tokens, "float32") / ratio)
    limited_grid = tf.stack([limited_height, limited_height * ratio], axis=0)
    ideal_grid = tf.where(
        ideal_grid[0] * ideal_grid[1] > max_tokens,
        limited_grid,
        ideal_grid,
    )

    lower_grid = tf.floor(ideal_grid)
    upper_grid = tf.math.ceil(ideal_grid)
    candidates = tf.stack(
        [
            tf.stack([lower_grid[0], lower_grid[1]]),
            tf.stack([lower_grid[0], upper_grid[1]]),
            tf.stack([upper_grid[0], lower_grid[1]]),
            tf.stack([upper_grid[0], upper_grid[1]]),
        ]
    )
    valid = (
        (candidates[:, 0] >= 1)
        & (candidates[:, 1] >= 1)
        & (candidates[:, 0] * candidates[:, 1] <= max_tokens)
    )
    aspect_error = tf.abs(
        candidates[:, 0] / candidates[:, 1]
        - tf.cast(height, "float32") / tf.cast(width, "float32")
    )
    aspect_error = tf.where(valid, aspect_error, 1e9)
    selected = tf.gather(candidates, tf.argmin(aspect_error))
    fallback = tf.maximum(tf.round(ideal_grid), 1)
    selected = tf.where(tf.reduce_any(valid), selected, fallback)
    selected = tf.cast(selected * resize_patch_size, "int32")
    return selected[0], selected[1]


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
    )
    return images[0] if rank == 3 else images


def _resize_pixels(
    images, orig_h, orig_w, target_h, target_w, method, antialias
):
    """Resize images with Keras ops to float32 values in `[0, 255]`."""
    if keras.backend.is_int_dtype(images.dtype):
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
    if input_is_integer:
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


def _normalize(images, image_converter, scale, offset, compute_dtype):
    """Apply `scale` and `offset` with the helpers of `image_converter`."""
    if scale is not None:
        scale = image_converter._expand_non_channel_dims(scale, images)
        images, scale = image_converter._convert_types(
            images, scale, compute_dtype
        )
        images = images * scale
    if offset is not None:
        offset = image_converter._expand_non_channel_dims(offset, images)
        images, offset = image_converter._convert_types(
            images, offset, images.dtype
        )
        images = images + offset
    return images


@keras_hub_export("keras_hub.layers.MuseGlimmerImageConverter")
class MuseGlimmerImageConverter(ImageConverter):
    """Image preprocessor for MuseGlimmer.

    Resizes an image to a patch-grid-aligned size, extracts
    `patch_size x patch_size` patches, duplicates each along the temporal
    axis (`patch_temporal`, since a still image has no motion), and
    flattens each patch to the vector `MuseGlimmerVisionPatchEmbedder`
    expects.

    Args:
        patch_size: int. Spatial patch size in pixels. Defaults to `14`.
        patch_temporal: int. Temporal patch size. Defaults to `2`.
        merge_size: int. Spatial merge (pixel-shuffle) factor. Defaults to
            `2`.
        max_image_tokens: int. Maximum number of merged vision tokens per
            image, used to derive the pixel budget. Defaults to `4096`.
        min_pixels: int. Minimum pixel budget. Defaults to the area of one
            merge-aligned patch stride squared.
    """

    backbone_cls = MuseGlimmerBackbone

    def __init__(
        self,
        patch_size=14,
        patch_temporal=2,
        merge_size=2,
        max_image_tokens=4096,
        min_pixels=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.patch_temporal = patch_temporal
        self.merge_size = merge_size
        self.max_image_tokens = max_image_tokens
        self._patch_stride = patch_size * merge_size
        self.min_pixels = min_pixels or self._patch_stride**2
        self.max_pixels = max_image_tokens * (self._patch_stride**2)

    @preprocessing_function
    def call(self, inputs):
        if in_tf_function():
            return self._call_tf(inputs)
        return self._call_ops(inputs)

    def _call_tf(self, inputs):
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
            image, self, self.scale, self.offset, self.compute_dtype
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
        image = inputs
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
            image, self, self.scale, self.offset, self.compute_dtype
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
                "min_pixels": self.min_pixels,
            }
        )
        return config
