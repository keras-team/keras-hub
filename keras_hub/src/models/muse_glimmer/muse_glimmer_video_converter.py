from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.video_converter import VideoConverter
from keras_hub.src.models.muse_glimmer.muse_glimmer_backbone import (
    MuseGlimmerBackbone,
)
from keras_hub.src.utils.tensor_utils import in_tf_function
from keras_hub.src.utils.tensor_utils import preprocessing_function

try:
    import tensorflow as tf
except ImportError:
    tf = None


@keras_hub_export("keras_hub.layers.MuseGlimmerVideoConverter")
class MuseGlimmerVideoConverter(VideoConverter):
    """Video preprocessor for MuseGlimmer.

    Per the model card, video is not handled by a distinct encoder — frames
    are sampled at `fps` (up to `num_frames`) and each sampled frame is
    processed through the same patch/merge pipeline as a still image (see
    HF's `get_video_features` == `get_image_features` pass-through).

    Args:
        patch_size: int. Spatial patch size in pixels. Defaults to `14`.
        patch_temporal: int. Temporal patch size (frames grouped per
            temporal patch). Defaults to `2`.
        merge_size: int. Spatial merge factor. Defaults to `2`.
        fps: float. Target sampling rate for `do_sample_frames`. Defaults
            to `2.0`.
        num_frames: int. Maximum number of sampled frames. Defaults to
            `96`.
        max_video_frame_tokens: int. Maximum merged vision tokens per
            frame, used to derive the pixel budget. Defaults to `144`.
    """

    backbone_cls = MuseGlimmerBackbone

    def __init__(
        self,
        patch_size=14,
        patch_temporal=2,
        merge_size=2,
        fps=2.0,
        num_frames=96,
        max_video_frame_tokens=144,
        interpolation="bilinear",
        antialias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.patch_temporal = patch_temporal
        self.merge_size = merge_size
        self.fps = fps
        self.num_frames = num_frames
        self.max_video_frame_tokens = max_video_frame_tokens
        # `VideoConverter` (unlike `ImageConverter`) doesn't expose these
        # directly — it delegates per-frame resizing to a composed
        # `image_converter`, but this class resizes frames itself in
        # `_call_ops`/`_call_tf`, so they're set here to match
        # `ImageConverter`'s own defaults.
        self.interpolation = interpolation
        self.antialias = antialias
        self._patch_stride = patch_size * merge_size
        self.max_pixels = max_video_frame_tokens * (self._patch_stride**2)
        self.min_pixels = self._patch_stride**2

    def _normalize(self, video):
        # `VideoConverter` (unlike `ImageConverter`) doesn't define
        # `_expand_non_channel_dims`/`_convert_types` itself — reuse them
        # from the composed `self.image_converter`, which shares the same
        # `scale`/`offset`/`data_format`.
        if self.scale is not None:
            scale = self.image_converter._expand_non_channel_dims(
                self.scale, video
            )
            video, scale = self.image_converter._convert_types(
                video, scale, self.compute_dtype
            )
            video = video * scale
        if self.offset is not None:
            offset = self.image_converter._expand_non_channel_dims(
                self.offset, video
            )
            video, offset = self.image_converter._convert_types(
                video, offset, video.dtype
            )
            video = video + offset
        return video

    @preprocessing_function
    def call(self, inputs):
        if in_tf_function():
            return self._call_tf(inputs)
        return self._call_ops(inputs)

    def _sample_frame_indices_tf(self, frame_count, source_fps):
        stride = tf.maximum(
            tf.cast(tf.round(source_fps / self.fps), "int32"), 1
        )
        indices = tf.range(0, frame_count, stride)
        indices = indices[: self.num_frames]
        return indices

    def _call_tf(self, inputs):
        video = tf.cast(inputs, "float32")
        frame_count = tf.shape(video)[0]
        # Assume the source is already at `self.fps`-equivalent sampling
        # when no metadata is available; callers that need exact-fps
        # subsampling should pre-sample before calling this converter.
        indices = self._sample_frame_indices_tf(
            frame_count, tf.cast(self.fps, "float32")
        )
        video = tf.gather(video, indices, axis=0)

        orig_h, orig_w = tf.shape(video)[1], tf.shape(video)[2]
        stride = tf.cast(self._patch_stride, "float32")
        total_pixels = tf.cast(orig_h * orig_w, "float32")
        max_pixels = tf.cast(self.max_pixels, "float32")
        scale = tf.minimum(
            1.0, tf.sqrt(max_pixels / tf.maximum(total_pixels, 1.0))
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
        video = tf.image.resize(
            video,
            (target_h, target_w),
            method=self.interpolation,
            antialias=self.antialias,
        )
        video = tf.clip_by_value(video, 0.0, 255.0)
        video = self._normalize(video)

        new_frame_count = tf.shape(video)[0]
        remainder = new_frame_count % self.patch_temporal
        pad_len = tf.where(remainder > 0, self.patch_temporal - remainder, 0)
        video = tf.cond(
            pad_len > 0,
            lambda: tf.concat(
                [video, tf.tile(video[-1:], [pad_len, 1, 1, 1])], axis=0
            ),
            lambda: video,
        )

        grid_t = tf.shape(video)[0] // self.patch_temporal
        grid_h, grid_w = (
            target_h // self.patch_size,
            target_w // self.patch_size,
        )
        video = tf.reshape(
            video,
            (
                grid_t,
                self.patch_temporal,
                grid_h,
                self.patch_size,
                grid_w,
                self.patch_size,
                3,
            ),
        )
        video = tf.transpose(video, (0, 2, 4, 1, 3, 5, 6))
        num_patches = grid_t * grid_h * grid_w
        patches = tf.reshape(
            video,
            (
                num_patches,
                self.patch_temporal * self.patch_size * self.patch_size * 3,
            ),
        )
        grid_thw = tf.stack([grid_t, grid_h, grid_w])
        return {"patches": patches, "grid_thw": grid_thw}

    def _call_ops(self, inputs):
        video = ops.cast(inputs, "float32")
        frame_count = int(ops.shape(video)[0])
        # Assumes a same-rate source (see `_call_tf` docstring note on
        # pre-sampling); simply caps the frame count at `num_frames`.
        indices = list(range(0, frame_count))[: self.num_frames]
        video = ops.take(video, indices, axis=0)

        orig_h, orig_w = int(ops.shape(video)[1]), int(ops.shape(video)[2])
        patch_stride = self._patch_stride
        total_pixels = float(orig_h * orig_w)
        scale = min(1.0, (self.max_pixels / total_pixels) ** 0.5)
        target_h = max(
            round(orig_h * scale / patch_stride) * patch_stride,
            patch_stride,
        )
        target_w = max(
            round(orig_w * scale / patch_stride) * patch_stride,
            patch_stride,
        )
        video = ops.image.resize(
            video,
            size=(target_h, target_w),
            interpolation=self.interpolation,
            antialias=self.antialias,
        )
        video = ops.clip(video, 0.0, 255.0)
        video = self._normalize(video)

        new_frame_count = int(ops.shape(video)[0])
        remainder = new_frame_count % self.patch_temporal
        if remainder > 0:
            pad_len = self.patch_temporal - remainder
            video = ops.concatenate(
                [video, ops.tile(video[-1:], (pad_len, 1, 1, 1))], axis=0
            )

        grid_t = int(ops.shape(video)[0]) // self.patch_temporal
        grid_h, grid_w = (
            target_h // self.patch_size,
            target_w // self.patch_size,
        )
        video = ops.reshape(
            video,
            (
                grid_t,
                self.patch_temporal,
                grid_h,
                self.patch_size,
                grid_w,
                self.patch_size,
                3,
            ),
        )
        video = ops.transpose(video, (0, 2, 4, 1, 3, 5, 6))
        num_patches = grid_t * grid_h * grid_w
        patches = ops.reshape(
            video,
            (
                num_patches,
                self.patch_temporal * self.patch_size * self.patch_size * 3,
            ),
        )
        grid_thw = ops.stack(
            [
                ops.array(grid_t, dtype="int32"),
                ops.array(grid_h, dtype="int32"),
                ops.array(grid_w, dtype="int32"),
            ]
        )
        return {"patches": patches, "grid_thw": grid_thw}

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "patch_size": self.patch_size,
                "patch_temporal": self.patch_temporal,
                "merge_size": self.merge_size,
                "fps": self.fps,
                "num_frames": self.num_frames,
                "max_video_frame_tokens": self.max_video_frame_tokens,
                "interpolation": self.interpolation,
                "antialias": self.antialias,
            }
        )
        return config
