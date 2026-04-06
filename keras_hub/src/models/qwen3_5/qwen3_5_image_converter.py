import math

import tensorflow as tf
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.qwen3_5.qwen3_5_backbone import Qwen3_5Backbone
from keras_hub.src.utils.tensor_utils import in_tf_function
from keras_hub.src.utils.tensor_utils import preprocessing_function


def _compute_target_size(h, w, min_pixels, max_pixels, patch_stride):
    """Compute target (H, W) respecting pixel budget and patch alignment.

    Args:
        h, w: int. Original image height/width in pixels.
        min_pixels: int. `shortest_edge` pixel count from config.
        max_pixels: int. `longest_edge` pixel count from config.
        patch_stride: int. patch_size * spatial_merge_size — output dims
            must be divisible by this.
    Returns:
        (target_h, target_w): ints.
    """
    total_pixels = h * w
    # Scale up/down to satisfy pixel count constraints.
    if total_pixels < min_pixels:
        scale = math.sqrt(min_pixels / total_pixels)
        h = math.ceil(h * scale)
        w = math.ceil(w * scale)
    elif total_pixels > max_pixels:
        scale = math.sqrt(max_pixels / total_pixels)
        h = math.floor(h * scale)
        w = math.floor(w * scale)

    # Round to nearest multiple of patch_stride.
    target_h = max(round(h / patch_stride) * patch_stride, patch_stride)
    target_w = max(round(w / patch_stride) * patch_stride, patch_stride)
    return target_h, target_w


@keras_hub_export("keras_hub.models.Qwen3_5ImageConverter")
class Qwen3_5ImageConverter(ImageConverter):
    """Image pre-processor for Qwen3.5-VL (image-only v1).

    Converts images to the patch tensor format expected by
    `Qwen3_5VisionEncoder` and also returns `grid_thw` metadata.

    Args:
        patch_size: int. Spatial size of each patch in pixels. Default 16.
        temporal_patch_size: int. Temporal patch size. For images this is
            always 1 (a single frame). Default 2 (matches HF config).
        spatial_merge_size: int. Spatial merge downsampling factor. Default 2.
        min_pixels: int. Minimum pixel budget for the resized image.
            Images smaller than this will be upscaled. Default 65536
            (= 256×256, from HF preprocessor_config.json `shortest_edge`).
        max_pixels: int. Maximum pixel budget. Images larger than this will
            be downscaled. Default 16777216 (= 4096×4096, `longest_edge`).
        image_mean: list[float]. Per-channel mean for normalisation.
            Default [0.5, 0.5, 0.5].
        image_std: list[float]. Per-channel std for normalisation.
            Default [0.5, 0.5, 0.5].
    """

    backbone_cls = Qwen3_5Backbone

    def __init__(
        self,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        min_pixels=65536,
        max_pixels=16777216,
        image_mean=None,
        image_std=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.image_mean = image_mean or [0.5, 0.5, 0.5]
        self.image_std = image_std or [0.5, 0.5, 0.5]
        # Patch stride: dimensions must be divisible by this.
        self._patch_stride = patch_size * spatial_merge_size

    @preprocessing_function
    def call(self, inputs):
        """Convert a single image to patch tensor + grid_thw.

        Args:
            inputs: uint8 or float32 tensor (H, W, 3) with pixel values in
                [0, 255].
        Returns:
            dict with:
                "patches": float32 tensor
                    (num_patches, temporal_patch_size,
                     patch_size, patch_size, 3) ready to feed
                     into Qwen3_5VisionEncoder.
                "grid_thw": int32 tensor [T=1, H//patch_size, W//patch_size]
                    (number of patches along each spatial axis).
        """
        if in_tf_function():
            return self._call_tf(inputs)
        else:
            return self._call_ops(inputs)

    def _call_tf(self, inputs):
        """TF graph-mode path (used inside tf.data.Dataset.map)."""
        image = tf.cast(inputs, "float32")
        orig_h = tf.shape(image)[0]
        orig_w = tf.shape(image)[1]

        # Compute tight pixel budget.
        total_pixels = tf.cast(orig_h * orig_w, "float32")
        min_pix = tf.cast(self.min_pixels, "float32")
        max_pix = tf.cast(self.max_pixels, "float32")
        stride = tf.cast(self._patch_stride, "float32")

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

        # Resize with bicubic to exact target dims.
        image = tf.image.resize(
            image[tf.newaxis],
            (target_h, target_w),
            method=tf.image.ResizeMethod.BICUBIC,
            antialias=True,
        )[0]
        image = tf.clip_by_value(image, 0.0, 255.0)

        # Normalise to [-1, 1].
        mean = tf.constant(self.image_mean, dtype="float32") * 255.0
        std = tf.constant(self.image_std, dtype="float32") * 255.0
        image = (image - mean) / std  # (H, W, 3)

        # Grid metadata.
        grid_h = target_h // self.patch_size
        grid_w = target_w // self.patch_size

        # Extract patches.
        image = tf.reshape(
            image,
            (grid_h, self.patch_size, grid_w, self.patch_size, 3),
        )
        image = tf.transpose(image, (0, 2, 1, 3, 4))
        num_patches = grid_h * grid_w
        image = tf.reshape(
            image, (num_patches, self.patch_size, self.patch_size, 3)
        )

        # Duplicate across temporal axis.
        image = tf.tile(
            tf.expand_dims(image, 1), [1, self.temporal_patch_size, 1, 1, 1]
        )

        grid_thw = tf.stack(
            [
                tf.constant(1, dtype="int32"),  # T=1 for images
                grid_h,
                grid_w,
            ]
        )

        return {"patches": image, "grid_thw": grid_thw}

    def _call_ops(self, inputs):
        """Backend-agnostic eager path (works with TF, JAX, PyTorch)."""
        image = ops.cast(inputs, "float32")
        orig_h = ops.shape(image)[0]
        orig_w = ops.shape(image)[1]

        # Compute tight pixel budget.
        total_pixels = ops.cast(orig_h * orig_w, "float32")
        min_pix = float(self.min_pixels)
        max_pix = float(self.max_pixels)
        stride = float(self._patch_stride)

        # In eager mode, total_pixels is a concrete scalar so Python
        # conditionals work correctly.
        if float(total_pixels) < min_pix:
            scale = ops.sqrt(min_pix / total_pixels)
        elif float(total_pixels) > max_pix:
            scale = ops.sqrt(max_pix / total_pixels)
        else:
            scale = 1.0

        target_h = ops.cast(
            ops.maximum(
                ops.round(ops.cast(orig_h, "float32") * scale / stride)
                * stride,
                stride,
            ),
            "int32",
        )
        target_w = ops.cast(
            ops.maximum(
                ops.round(ops.cast(orig_w, "float32") * scale / stride)
                * stride,
                stride,
            ),
            "int32",
        )

        # Resize with bicubic to exact target dims.
        image = ops.image.resize(
            ops.expand_dims(image, 0),
            size=(target_h, target_w),
            interpolation="bicubic",
            antialias=True,
        )[0]
        image = ops.clip(image, 0.0, 255.0)

        # Normalise to [-1, 1].
        mean = ops.array(self.image_mean, dtype="float32") * 255.0
        std = ops.array(self.image_std, dtype="float32") * 255.0
        image = (image - mean) / std  # (H, W, 3)

        # Grid metadata.
        grid_h = target_h // self.patch_size
        grid_w = target_w // self.patch_size

        # Extract patches.
        image = ops.reshape(
            image,
            (grid_h, self.patch_size, grid_w, self.patch_size, 3),
        )
        image = ops.transpose(image, (0, 2, 1, 3, 4))
        num_patches = grid_h * grid_w
        image = ops.reshape(
            image, (num_patches, self.patch_size, self.patch_size, 3)
        )

        # Duplicate across temporal axis.
        image = ops.tile(
            ops.expand_dims(image, 1), [1, self.temporal_patch_size, 1, 1, 1]
        )

        grid_thw = ops.stack(
            [
                ops.array(1, dtype="int32"),  # T=1 for images
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
                "image_mean": self.image_mean,
                "image_std": self.image_std,
            }
        )
        return config
