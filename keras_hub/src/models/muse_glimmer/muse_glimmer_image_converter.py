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
        image = tf.cast(inputs, "float32")
        orig_h, orig_w = tf.shape(image)[0], tf.shape(image)[1]
        total_pixels = tf.cast(orig_h * orig_w, "float32")
        stride = tf.cast(self._patch_stride, "float32")
        min_pix = tf.cast(self.min_pixels, "float32")
        max_pix = tf.cast(self.max_pixels, "float32")

        scale = tf.cond(
            total_pixels < min_pix,
            lambda: tf.sqrt(min_pix / total_pixels),
            lambda: tf.cond(
                total_pixels > max_pix,
                lambda: tf.sqrt(max_pix / total_pixels),
                lambda: tf.constant(1.0),
            ),
        )
        target_h = tf.cast(
            tf.maximum(
                tf.round(tf.cast(orig_h, "float32") * scale / stride) * stride,
                stride,
            ),
            "int32",
        )
        target_w = tf.cast(
            tf.maximum(
                tf.round(tf.cast(orig_w, "float32") * scale / stride) * stride,
                stride,
            ),
            "int32",
        )
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
        image = tf.transpose(image, (0, 2, 1, 3, 4))
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
        image = ops.cast(inputs, "float32")
        orig_h, orig_w = ops.shape(image)[0], ops.shape(image)[1]
        total_pixels = float(ops.cast(orig_h * orig_w, "float32"))
        stride = float(self._patch_stride)

        if total_pixels < self.min_pixels:
            scale = (self.min_pixels / total_pixels) ** 0.5
        elif total_pixels > self.max_pixels:
            scale = (self.max_pixels / total_pixels) ** 0.5
        else:
            scale = 1.0

        target_h = max(
            round(int(orig_h) * scale / stride) * self._patch_stride,
            self._patch_stride,
        )
        target_w = max(
            round(int(orig_w) * scale / stride) * self._patch_stride,
            self._patch_stride,
        )
        image = ops.image.resize(
            ops.expand_dims(image, 0),
            size=(target_h, target_w),
            interpolation=self.interpolation,
            antialias=self.antialias,
        )[0]
        image = ops.clip(image, 0.0, 255.0)
        image = self._normalize(image)

        grid_h, grid_w = (
            target_h // self.patch_size,
            target_w // self.patch_size,
        )
        image = ops.reshape(
            image, (grid_h, self.patch_size, grid_w, self.patch_size, 3)
        )
        image = ops.transpose(image, (0, 2, 1, 3, 4))
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
