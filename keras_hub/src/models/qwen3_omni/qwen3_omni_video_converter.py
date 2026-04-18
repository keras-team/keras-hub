import math

import tensorflow as tf
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.video_converter import VideoConverter
from keras_hub.src.models.qwen3_omni.qwen3_omni_backbone import (
    Qwen3OmniBackbone,
)
from keras_hub.src.utils.tensor_utils import in_tf_function
from keras_hub.src.utils.tensor_utils import preprocessing_function


@keras_hub_export("keras_hub.layers.Qwen3OmniVideoConverter")
class Qwen3OmniVideoConverter(VideoConverter):
    """Video pre-processor for Qwen3-Omni.

    Converts videos to the patch tensor format expected by
    ``Qwen3OmniVisionEncoder`` and also returns ``grid_thw`` metadata.

    Args:
        patch_size: int. Spatial size of each patch in pixels. Default ``16``.
        temporal_patch_size: int. Frames grouped per temporal patch.
            Default ``2``.
        spatial_merge_size: int. Spatial merge downsampling factor.
            Default ``2``.
        min_pixels: int. Minimum pixel budget for the resized frames.
            Frames smaller than this will be upscaled. Default ``65536``.
        max_pixels: int. Maximum pixel budget. Frames larger than this will
            be downscaled. Default ``16777216`` (= 4096 x 4096).
        interpolation: str. Interpolation method for resizing.
            Defaults to ``"bicubic"``.
        antialias: bool. Whether to apply antialiasing when resizing.
            Defaults to ``True``.
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
        interpolation="bicubic",
        antialias=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.interpolation = interpolation
        self.antialias = antialias
        self._patch_stride = patch_size * spatial_merge_size

    @preprocessing_function
    def call(self, inputs):
        """Convert a single video to patch tensor + grid_thw.

        Args:
            inputs: uint8 or float32 tensor ``(T, H, W, 3)`` with pixel
                values in ``[0, 255]``.

        Returns:
            dict with keys ``"patches"`` and ``"grid_thw"``.
        """
        if in_tf_function():
            return self._call_tf(inputs)
        else:
            return self._call_ops(inputs)

    def _call_tf(self, inputs):
        """TF graph-mode path (used inside ``tf.data.Dataset.map``)."""
        video = tf.cast(inputs, "float32")
        frame_count = tf.shape(video)[0]
        orig_h = tf.shape(video)[1]
        orig_w = tf.shape(video)[2]

        stride = tf.cast(self._patch_stride, "float32")
        stride_int = tf.cast(self._patch_stride, "int32")

        h_bar = tf.cast(
            tf.maximum(
                tf.round(tf.cast(orig_h, "float32") / stride) * stride,
                stride,
            ),
            "int32",
        )
        w_bar = tf.cast(
            tf.maximum(
                tf.round(tf.cast(orig_w, "float32") / stride) * stride,
                stride,
            ),
            "int32",
        )
        tps = self.temporal_patch_size
        t_bar = ((frame_count + tps - 1) // tps) * tps

        total_thw = tf.cast(t_bar * h_bar * w_bar, "float32")
        orig_thw = tf.cast(frame_count * orig_h * orig_w, "float32")
        min_pix = tf.cast(self.min_pixels, "float32")
        max_pix = tf.cast(self.max_pixels, "float32")

        def _scale_down():
            beta = tf.sqrt(orig_thw / max_pix)
            new_h = tf.maximum(
                stride_int,
                tf.cast(
                    tf.floor(tf.cast(orig_h, "float32") / beta / stride)
                    * stride,
                    "int32",
                ),
            )
            new_w = tf.maximum(
                stride_int,
                tf.cast(
                    tf.floor(tf.cast(orig_w, "float32") / beta / stride)
                    * stride,
                    "int32",
                ),
            )
            return new_h, new_w

        def _scale_up():
            beta = tf.sqrt(min_pix / orig_thw)
            new_h = tf.cast(
                tf.math.ceil(tf.cast(orig_h, "float32") * beta / stride)
                * stride,
                "int32",
            )
            new_w = tf.cast(
                tf.math.ceil(tf.cast(orig_w, "float32") * beta / stride)
                * stride,
                "int32",
            )
            return new_h, new_w

        def _no_scale():
            return h_bar, w_bar

        target_h, target_w = tf.cond(
            total_thw > max_pix,
            _scale_down,
            lambda: tf.cond(total_thw < min_pix, _scale_up, _no_scale),
        )

        video = tf.image.resize(
            video,
            (target_h, target_w),
            method=self.interpolation,
            antialias=self.antialias,
        )
        video = tf.clip_by_value(video, 0.0, 255.0)

        if self.scale is not None:
            scale = tf.constant(self.scale, dtype="float32")
            scale = tf.reshape(scale, (1, 1, 1, -1))
            video = tf.cast(video, "float32") * scale
        if self.offset is not None:
            offset = tf.constant(self.offset, dtype="float32")
            offset = tf.reshape(offset, (1, 1, 1, -1))
            video = video + offset

        remainder = frame_count % self.temporal_patch_size
        pad_len = tf.cond(
            remainder > 0,
            lambda: self.temporal_patch_size - remainder,
            lambda: tf.constant(0, dtype="int32"),
        )
        padded_video = tf.cond(
            pad_len > 0,
            lambda: tf.concat(
                [video, tf.tile(video[-1:], [pad_len, 1, 1, 1])], axis=0
            ),
            lambda: video,
        )
        new_frame_count = tf.shape(padded_video)[0]

        grid_t = new_frame_count // self.temporal_patch_size
        grid_h = target_h // self.patch_size
        grid_w = target_w // self.patch_size

        video_reshaped = tf.reshape(
            padded_video,
            (
                grid_t,
                self.temporal_patch_size,
                grid_h,
                self.patch_size,
                grid_w,
                self.patch_size,
                3,
            ),
        )
        video_transposed = tf.transpose(video_reshaped, (0, 2, 4, 1, 3, 5, 6))
        total_patches = grid_t * grid_h * grid_w
        patches = tf.reshape(
            video_transposed,
            (
                total_patches,
                self.temporal_patch_size,
                self.patch_size,
                self.patch_size,
                3,
            ),
        )

        grid_thw = tf.stack([grid_t, grid_h, grid_w])

        return {"patches": patches, "grid_thw": grid_thw}

    def _call_ops(self, inputs):
        """Backend-agnostic eager path."""
        video = ops.cast(inputs, "float32")
        frame_count = int(ops.shape(video)[0])
        orig_h = int(ops.shape(video)[1])
        orig_w = int(ops.shape(video)[2])

        stride = self._patch_stride

        h_bar = max(round(orig_h / stride) * stride, stride)
        w_bar = max(round(orig_w / stride) * stride, stride)
        t_bar = (
            math.ceil(frame_count / self.temporal_patch_size)
            * self.temporal_patch_size
        )

        if t_bar * h_bar * w_bar > self.max_pixels:
            beta = math.sqrt((frame_count * orig_h * orig_w) / self.max_pixels)
            h_bar = max(
                stride,
                math.floor(orig_h / beta / stride) * stride,
            )
            w_bar = max(
                stride,
                math.floor(orig_w / beta / stride) * stride,
            )
        elif t_bar * h_bar * w_bar < self.min_pixels:
            beta = math.sqrt(self.min_pixels / (frame_count * orig_h * orig_w))
            h_bar = math.ceil(orig_h * beta / stride) * stride
            w_bar = math.ceil(orig_w * beta / stride) * stride

        target_h = h_bar
        target_w = w_bar

        video = ops.image.resize(
            video,
            size=(target_h, target_w),
            interpolation=self.interpolation,
            antialias=self.antialias,
        )
        video = ops.clip(video, 0.0, 255.0)

        if self.scale is not None:
            scale = ops.reshape(
                ops.convert_to_tensor(self.scale, dtype="float32"),
                (1, 1, 1, -1),
            )
            video = ops.cast(video, "float32") * scale
        if self.offset is not None:
            offset = ops.reshape(
                ops.convert_to_tensor(self.offset, dtype="float32"),
                (1, 1, 1, -1),
            )
            video = video + offset

        remainder = frame_count % self.temporal_patch_size
        if remainder > 0:
            pad_len = self.temporal_patch_size - remainder
            last_frame = video[-1:]
            last_frame_tiled = ops.tile(last_frame, [pad_len, 1, 1, 1])
            video = ops.concatenate([video, last_frame_tiled], axis=0)

        new_frame_count = int(ops.shape(video)[0])
        grid_t = new_frame_count // self.temporal_patch_size
        grid_h = target_h // self.patch_size
        grid_w = target_w // self.patch_size

        video_reshaped = ops.reshape(
            video,
            (
                grid_t,
                self.temporal_patch_size,
                grid_h,
                self.patch_size,
                grid_w,
                self.patch_size,
                3,
            ),
        )
        video_transposed = ops.transpose(video_reshaped, (0, 2, 4, 1, 3, 5, 6))
        total_patches = grid_t * grid_h * grid_w
        patches = ops.reshape(
            video_transposed,
            (
                total_patches,
                self.temporal_patch_size,
                self.patch_size,
                self.patch_size,
                3,
            ),
        )

        grid_thw = ops.stack(
            [
                ops.cast(grid_t, "int32"),
                ops.cast(grid_h, "int32"),
                ops.cast(grid_w, "int32"),
            ]
        )

        return {"patches": patches, "grid_thw": grid_thw}

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "temporal_patch_size": self.temporal_patch_size,
                "spatial_merge_size": self.spatial_merge_size,
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels,
                "interpolation": self.interpolation,
                "antialias": self.antialias,
            }
        )
        return config
