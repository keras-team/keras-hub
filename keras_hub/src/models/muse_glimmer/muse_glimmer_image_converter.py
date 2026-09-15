import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.muse_glimmer.muse_glimmer_backbone import (
    MuseGlimmerBackbone,
)
from keras_hub.src.utils.tensor_utils import in_tf_function
from keras_hub.src.utils.tensor_utils import preprocessing_function

try:
    import tensorflow as tf
except ImportError:
    tf = None


def _smart_resize(height, width, patch_size, merge_size, max_tokens):
    """Select a patch grid that preserves the input aspect ratio."""
    resize_patch_size = patch_size * merge_size
    ideal_grid = ops.array(
        [height / resize_patch_size, width / resize_patch_size],
        dtype="float32",
    )
    ratio = ideal_grid[1] / ideal_grid[0]
    limited_height = ops.sqrt(max_tokens / ratio)
    limited_grid = ops.stack([limited_height, limited_height * ratio], axis=0)
    ideal_grid = ops.where(
        ideal_grid[0] * ideal_grid[1] > max_tokens,
        limited_grid,
        ideal_grid,
    )

    lower_grid = ops.floor(ideal_grid)
    upper_grid = ops.ceil(ideal_grid)
    candidates = ops.stack(
        [
            ops.stack([lower_grid[0], lower_grid[1]]),
            ops.stack([lower_grid[0], upper_grid[1]]),
            ops.stack([upper_grid[0], lower_grid[1]]),
            ops.stack([upper_grid[0], upper_grid[1]]),
        ]
    )
    valid = (
        (candidates[:, 0] >= 1)
        & (candidates[:, 1] >= 1)
        & (candidates[:, 0] * candidates[:, 1] <= max_tokens)
    )
    aspect_error = ops.abs(candidates[:, 0] / candidates[:, 1] - height / width)
    aspect_error = ops.where(valid, aspect_error, 1e9)
    selected = ops.take(candidates, ops.argmin(aspect_error), axis=0)
    fallback = ops.maximum(ops.round(ideal_grid), 1)
    selected = ops.where(ops.any(valid), selected, fallback)
    selected = ops.convert_to_numpy(selected)
    return (
        int(selected[0]) * resize_patch_size,
        int(selected[1]) * resize_patch_size,
    )


def _smart_resize_tf(height, width, patch_size, merge_size, max_tokens):
    """Select a patch grid inside a TensorFlow graph."""
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
    upper_grid = tf.ceil(ideal_grid)
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

    def _normalize(self, image):
        if self.scale is not None:
            scale = self._expand_non_channel_dims(self.scale, image)
            image, scale = self._convert_types(image, scale, self.compute_dtype)
            image = image * scale
        if self.offset is not None:
            offset = self._expand_non_channel_dims(self.offset, image)
            image, offset = self._convert_types(image, offset, image.dtype)
            image = image + offset
        return image

    def _call_tf(self, inputs):
        input_is_integer = tf.as_dtype(inputs.dtype).is_integer
        image = tf.cast(inputs, "float32")
        orig_h, orig_w = tf.shape(image)[0], tf.shape(image)[1]
        target_h, target_w = _smart_resize_tf(
            orig_h,
            orig_w,
            self.patch_size,
            self.merge_size,
            self.max_image_tokens,
        )
        if input_is_integer:
            # Matches torchvision's separable uint8 resize: width pass,
            # round to uint8 range, then height pass, round again.
            image = tf.image.resize(
                image[tf.newaxis],
                (orig_h, target_w),
                method=self.interpolation,
                antialias=self.antialias,
            )[0]
            image = tf.round(tf.clip_by_value(image, 0.0, 255.0))
            image = tf.image.resize(
                image[tf.newaxis],
                (target_h, target_w),
                method=self.interpolation,
                antialias=self.antialias,
            )[0]
            image = tf.round(tf.clip_by_value(image, 0.0, 255.0))
        else:
            image = tf.image.resize(
                image[tf.newaxis],
                (target_h, target_w),
                method=self.interpolation,
                antialias=self.antialias,
            )[0]
        image = tf.clip_by_value(image, 0.0, 255.0)
        image = self._normalize(image)

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

    def _resize(self, image, orig_h, orig_w, target_h, target_w):
        # The PyTorch backend does not support Lanczos interpolation in
        # `ops.image.resize`. Fall back to `scale_and_translate`, the
        # primitive `resize` itself uses on other backends for this case.
        if keras.backend.backend() == "torch" and self.interpolation in (
            "lanczos3",
            "lanczos5",
        ):
            scale = ops.array(
                [target_h / orig_h, target_w / orig_w], dtype="float32"
            )
            return ops.image.scale_and_translate(
                image,
                (target_h, target_w, 3),
                scale=scale,
                translation=ops.zeros((2,), dtype="float32"),
                spatial_dims=(0, 1),
                method=self.interpolation,
                antialias=self.antialias,
            )
        return ops.image.resize(
            ops.expand_dims(image, 0),
            size=(target_h, target_w),
            interpolation=self.interpolation,
            antialias=self.antialias,
        )[0]

    def _call_ops(self, inputs):
        input_is_integer = keras.backend.is_int_dtype(inputs.dtype)
        image = inputs
        orig_h, orig_w = int(ops.shape(image)[0]), int(ops.shape(image)[1])
        target_h, target_w = _smart_resize(
            orig_h,
            orig_w,
            self.patch_size,
            self.merge_size,
            self.max_image_tokens,
        )
        if input_is_integer:
            # Matches torchvision's separable uint8 resize: width pass,
            # round to uint8 range, then height pass, round again.
            image = self._resize(image, orig_h, orig_w, orig_h, target_w)
            # The TF backend's `ops.image.resize` preserves integer
            # dtypes, so clip/round need a float cast first.
            image = ops.cast(image, "float32")
            image = ops.round(ops.clip(image, 0.0, 255.0))
            image = self._resize(image, orig_h, target_w, target_h, target_w)
            image = ops.round(ops.clip(image, 0.0, 255.0))
        else:
            image = self._resize(image, orig_h, orig_w, target_h, target_w)
        image = ops.cast(image, "float32")
        image = ops.clip(image, 0.0, 255.0)
        image = self._normalize(image)

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
