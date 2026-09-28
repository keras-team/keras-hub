from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.video_converter import VideoConverter
from keras_hub.src.models.muse_glimmer.muse_glimmer_backbone import (
    MuseGlimmerBackbone,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_image_converter import (
    _normalize,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_image_converter import (
    _resize_pixels,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_image_converter import (
    _resize_pixels_tf,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_image_converter import (
    _smart_resize,
)
from keras_hub.src.models.muse_glimmer.muse_glimmer_image_converter import (
    _smart_resize_tf,
)
from keras_hub.src.utils.tensor_utils import in_tf_function
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import tf


@keras_hub_export("keras_hub.layers.MuseGlimmerVideoConverter")
class MuseGlimmerVideoConverter(VideoConverter):
    """Video preprocessor for MuseGlimmer.

    Per the model card, video is not handled by a distinct encoder. Each
    frame goes through the same patch/merge pipeline as a still image (see
    HF's `get_video_features` == `get_image_features` pass-through).

    The converter does not resample frames. Pre-sample the video to the
    target frame rate before you call the converter. The converter keeps
    at most `num_frames` frames.

    Args:
        patch_size: int. Spatial patch size in pixels. Defaults to `14`.
        patch_temporal: int. Temporal patch size (frames grouped per
            temporal patch). Defaults to `2`.
        merge_size: int. Spatial merge factor. Defaults to `2`.
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

    @preprocessing_function
    def call(self, inputs):
        if in_tf_function():
            return self._call_tf(inputs)
        return self._call_ops(inputs)

    def _call_tf(self, inputs):
        input_is_integer = tf.as_dtype(inputs.dtype).is_integer
        video = tf.cast(inputs, "float32")[: self.num_frames]

        orig_h, orig_w = tf.shape(video)[1], tf.shape(video)[2]
        target_h, target_w = _smart_resize_tf(
            orig_h,
            orig_w,
            self.patch_size,
            self.merge_size,
            self.max_video_frame_tokens,
        )
        video = _resize_pixels_tf(
            video,
            input_is_integer,
            orig_h,
            target_h,
            target_w,
            self.interpolation,
            self.antialias,
        )
        # `VideoConverter` does not define `_expand_non_channel_dims` or
        # `_convert_types`. The composed `self.image_converter` defines
        # them and shares the same `data_format`.
        video = _normalize(
            video,
            self.image_converter,
            self.scale,
            self.offset,
            self.compute_dtype,
        )

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
        video = tf.transpose(video, (0, 2, 4, 1, 6, 3, 5))
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
        video = inputs[: self.num_frames]

        orig_h, orig_w = int(ops.shape(video)[1]), int(ops.shape(video)[2])
        target_h, target_w = _smart_resize(
            orig_h,
            orig_w,
            self.patch_size,
            self.merge_size,
            self.max_video_frame_tokens,
        )
        video = _resize_pixels(
            video,
            orig_h,
            orig_w,
            target_h,
            target_w,
            self.interpolation,
            self.antialias,
        )
        video = _normalize(
            video,
            self.image_converter,
            self.scale,
            self.offset,
            self.compute_dtype,
        )

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
        video = ops.transpose(video, (0, 2, 4, 1, 6, 3, 5))
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
                "num_frames": self.num_frames,
                "max_video_frame_tokens": self.max_video_frame_tokens,
                "interpolation": self.interpolation,
                "antialias": self.antialias,
            }
        )
        return config
