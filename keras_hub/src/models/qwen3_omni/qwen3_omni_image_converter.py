import math

import tensorflow as tf
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.qwen3_omni.qwen3_omni_backbone import (
    Qwen3OmniBackbone,
)
from keras_hub.src.utils.tensor_utils import in_tf_function
from keras_hub.src.utils.tensor_utils import preprocessing_function


def _compute_target_size(h, w, min_pixels, max_pixels, patch_stride):
    """Compute target (H, W) respecting pixel budget and patch alignment.

    Args:
        h: int. Original image height in pixels.
        w: int. Original image width in pixels.
        min_pixels: int. Minimum total pixel count.
        max_pixels: int. Maximum total pixel count.
        patch_stride: int. ``patch_size * spatial_merge_size`` — output dims
            must be divisible by this.

    Returns:
        Tuple ``(target_h, target_w)``.
    """
    total_pixels = h * w
    if total_pixels < min_pixels:
        scale = math.sqrt(min_pixels / total_pixels)
        h = math.ceil(h * scale)
        w = math.ceil(w * scale)
    elif total_pixels > max_pixels:
        scale = math.sqrt(max_pixels / total_pixels)
        h = math.floor(h * scale)
        w = math.floor(w * scale)

    target_h = max(round(h / patch_stride) * patch_stride, patch_stride)
    target_w = max(round(w / patch_stride) * patch_stride, patch_stride)
    return target_h, target_w


@keras_hub_export("keras_hub.layers.Qwen3OmniImageConverter")
class Qwen3OmniImageConverter(ImageConverter):
    """Image pre-processor for Qwen3-Omni.

    Converts images to the patch tensor format expected by
    ``Qwen3OmniVisionEncoder`` and also returns ``grid_thw`` metadata.

    Args:
        patch_size: int. Spatial size of each patch in pixels. Default ``16``.
        temporal_patch_size: int. Temporal patch size. For images this is
            always 1 (a single frame). Default ``2`` (matches HF config).
        spatial_merge_size: int. Spatial merge downsampling factor. Default
            ``2``.
        min_pixels: int. Minimum pixel budget for the resized image.
            Default ``65536`` (= 256 x 256).
        max_pixels: int. Maximum pixel budget. Default ``16777216``
            (= 4096 x 4096).
        scale: float or list of floats. Per-channel scale for normalisation.
        offset: float or list of floats. Per-channel offset for
            normalisation.
    """

    backbone_cls = Qwen3OmniBackbone

    def __init__(
        self,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        min_pixels=65536,
        max_pixels=16777216,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self._patch_stride = patch_size * spatial_merge_size

    @preprocessing_function
    def call(self, inputs):
        """Convert a single image to patch tensor + grid_thw.

        Args:
            inputs: uint8 or float32 tensor ``(H, W, 3)`` with pixel values
                in ``[0, 255]``.

        Returns:
            dict with keys ``"patches"`` and ``"grid_thw"``.
        """
        if in_tf_function():
            return self._call_tf(inputs)
        else:
            return self._call_ops(inputs)

    def _call_tf(self, inputs):
        """TF graph-mode path (used inside ``tf.data.Dataset.map``)."""
        image = tf.cast(inputs, "float32")
        orig_h = tf.shape(image)[0]
        orig_w = tf.shape(image)[1]

        total_pixels = tf.cast(orig_h * orig_w, "float32")
        min_pix = tf.cast(self.min_pixels, "float32")
        max_pix = tf.cast(self.max_pixels, "float32")
        stride = tf.cast(self._patch_stride, "float32")

        upscale = total_pixels < min_pix
        downscale = total_pixels > max_pix
        scale = tf.cond(
            upscale,
            lambda: tf.sqrt(min_pix / total_pixels),
            lambda: tf.cond(
                downscale,
                lambda: tf.sqrt(max_pix / total_pixels),
                lambda: tf.constant(1.0),
            ),
        )

        scaled_h = tf.cast(orig_h, "float32") * scale
        scaled_w = tf.cast(orig_w, "float32") * scale
        scaled_h = tf.where(
            upscale,
            tf.math.ceil(scaled_h),
            tf.where(downscale, tf.math.floor(scaled_h), scaled_h),
        )
        scaled_w = tf.where(
            upscale,
            tf.math.ceil(scaled_w),
            tf.where(downscale, tf.math.floor(scaled_w), scaled_w),
        )

        target_h = tf.cast(
            tf.maximum(tf.round(scaled_h / stride) * stride, stride),
            "int32",
        )
        target_w = tf.cast(
            tf.maximum(tf.round(scaled_w / stride) * stride, stride),
            "int32",
        )

        image = tf.image.resize(
            image[tf.newaxis],
            (target_h, target_w),
            method=self.interpolation,
            antialias=self.antialias,
        )[0]
        image = tf.clip_by_value(image, 0.0, 255.0)

        if self.scale is not None:
            scale_t = self._expand_non_channel_dims(self.scale, image)
            image, scale_t = self._convert_types(
                image, scale_t, self.compute_dtype
            )
            image = image * scale_t
        if self.offset is not None:
            offset_t = self._expand_non_channel_dims(self.offset, image)
            image, offset_t = self._convert_types(image, offset_t, image.dtype)
            image = image + offset_t

        grid_h = target_h // self.patch_size
        grid_w = target_w // self.patch_size

        image = tf.reshape(
            image,
            (grid_h, self.patch_size, grid_w, self.patch_size, 3),
        )
        image = tf.transpose(image, (0, 2, 1, 3, 4))
        num_patches = grid_h * grid_w
        image = tf.reshape(
            image, (num_patches, self.patch_size, self.patch_size, 3)
        )

        image = tf.tile(
            tf.expand_dims(image, 1),
            [1, self.temporal_patch_size, 1, 1, 1],
        )

        grid_thw = tf.stack([tf.constant(1, dtype="int32"), grid_h, grid_w])

        return {"patches": image, "grid_thw": grid_thw}

    def _call_ops(self, inputs):
        """Backend-agnostic eager path."""
        image = ops.cast(inputs, "float32")
        orig_h = ops.shape(image)[0]
        orig_w = ops.shape(image)[1]

        target_h, target_w = _compute_target_size(
            int(orig_h),
            int(orig_w),
            self.min_pixels,
            self.max_pixels,
            self._patch_stride,
        )

        image = ops.image.resize(
            ops.expand_dims(image, 0),
            size=(target_h, target_w),
            interpolation=self.interpolation,
            antialias=self.antialias,
        )[0]
        image = ops.clip(image, 0.0, 255.0)

        if self.scale is not None:
            scale = self._expand_non_channel_dims(self.scale, image)
            image, scale = self._convert_types(image, scale, self.compute_dtype)
            image = image * scale
        if self.offset is not None:
            offset = self._expand_non_channel_dims(self.offset, image)
            image, offset = self._convert_types(image, offset, image.dtype)
            image = image + offset

        grid_h = target_h // self.patch_size
        grid_w = target_w // self.patch_size

        image = ops.reshape(
            image,
            (grid_h, self.patch_size, grid_w, self.patch_size, 3),
        )
        image = ops.transpose(image, (0, 2, 1, 3, 4))
        num_patches = grid_h * grid_w
        image = ops.reshape(
            image, (num_patches, self.patch_size, self.patch_size, 3)
        )

        image = ops.tile(
            ops.expand_dims(image, 1),
            [1, self.temporal_patch_size, 1, 1, 1],
        )

        grid_thw = ops.stack(
            [
                ops.array(1, dtype="int32"),
                grid_h,
                grid_w,
            ]
        )

        return {"patches": image, "grid_thw": grid_thw}

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "temporal_patch_size": self.temporal_patch_size,
                "spatial_merge_size": self.spatial_merge_size,
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels,
            }
        )
        return config
