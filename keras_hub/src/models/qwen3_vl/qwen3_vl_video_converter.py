from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.video_converter import VideoConverter
from keras_hub.src.models.qwen3_vl.qwen3_vl_backbone import Qwen3_VLBackbone
from keras_hub.src.models.qwen3_vl.qwen3_vl_image_converter import (
    resize_integer_pixels,
)
from keras_hub.src.models.qwen3_vl.qwen3_vl_image_converter import (
    resize_integer_pixels_tf,
)
from keras_hub.src.models.qwen3_vl.qwen3_vl_image_converter import (
    smart_resize_volume as smart_resize,
)
from keras_hub.src.models.qwen3_vl.qwen3_vl_image_converter import (
    smart_resize_volume_tf as _smart_resize_tf,
)
from keras_hub.src.utils.tensor_utils import convert_preprocessing_outputs_grain
from keras_hub.src.utils.tensor_utils import in_grain_data_pipeline
from keras_hub.src.utils.tensor_utils import in_tf_function
from keras_hub.src.utils.tensor_utils import is_int_dtype
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import tf


@keras_hub_export("keras_hub.layers.Qwen3_VLVideoConverter")
class Qwen3_VLVideoConverter(VideoConverter):
    """Video converter for Qwen3-VL.

    Resizes the frames of a video with the Qwen `smart_resize` rule,
    normalizes them, and cuts them into the flattened patch tensor consumed by
    `Qwen3_VLVisionEncoder`. Patches are emitted in merge-block order, one
    temporal group after another. A video with an odd number of frames is
    padded by repeating its last frame.

    Frame selection happens before the call: use `sample_frame_indices` to
    pick which frames of a decoded video to pass in.

    Args:
        patch_size: int. Side of a square patch in pixels. Defaults to `16`.
        temporal_patch_size: int. Number of frames per patch. Defaults to `2`.
        spatial_merge_size: int. Side of the patch block merged into one
            token. Defaults to `2`.
        min_pixels: int. Videos whose `frames * height * width` is smaller
            than this are upscaled. Defaults to `4096` (`shortest_edge` in
            the HF video config).
        max_pixels: int. Videos above this volume are downscaled. Defaults to
            `25165824` (`longest_edge` in the HF video config).
        fps: float. Target sampling rate used by `sample_frame_indices`.
            Defaults to `2.0`.
        min_frames: int. Lower bound on the sampled frame count. Defaults to
            `4`.
        max_frames: int. Upper bound on the sampled frame count. Defaults to
            `768`.
        scale: float or list of floats. Multiplier applied to pixels in
            `[0, 255]`. Defaults to `2 / 255`.
        offset: float or list of floats. Value added after scaling. The
            defaults map pixels to `[-1, 1]`, which is the HF
            `mean=std=0.5` normalization. Defaults to `-1.0`.
        interpolation: string. Resize method. Defaults to `"bicubic"`.
        antialias: bool. Whether resizing is antialiased. Defaults to `True`.

    Call arguments:
        inputs: A `(frames, height, width, 3)` video with values in
            `[0, 255]`.

    Returns:
        A dict with `"patches"`, a float tensor of shape
        `(num_patches, temporal_patch_size, patch_size, patch_size, 3)`, and
        `"grid_thw"`, the int32 vector `[frames // temporal_patch_size,
        height // patch_size, width // patch_size]` after padding.
    """

    backbone_cls = Qwen3_VLBackbone

    def __init__(
        self,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        min_pixels=4096,
        max_pixels=25165824,
        fps=2.0,
        min_frames=4,
        max_frames=768,
        scale=2.0 / 255.0,
        offset=-1.0,
        interpolation="bicubic",
        antialias=True,
        **kwargs,
    ):
        super().__init__(scale=scale, offset=offset, **kwargs)
        # The base class builds an image converter with default resize
        # settings. This converter resizes and normalizes the frames itself.
        self.image_converter = None
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.fps = fps
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.interpolation = interpolation
        self.antialias = antialias
        self._factor = patch_size * spatial_merge_size

    def sample_frame_indices(self, total_frames, video_fps=24.0):
        """Choose which frames of a decoded video to keep.

        Mirrors the HF sampler: the frame count is `total_frames / video_fps
        * fps`, clipped to `[min_frames, max_frames]` and to `total_frames`,
        and the indices are spread evenly over the video.

        Args:
            total_frames: int. Number of frames in the source video.
            video_fps: float. Frame rate of the source video. Defaults to
                `24.0`, the HF fallback when the metadata has no rate.

        Returns:
            A 1D int array of frame indices.
        """
        num_frames = int(total_frames / video_fps * self.fps)
        num_frames = min(
            max(num_frames, self.min_frames), self.max_frames, total_frames
        )
        indices = ops.linspace(0.0, total_frames - 1.0, num_frames)
        return ops.cast(ops.round(indices), "int32")

    @preprocessing_function
    def call(self, inputs):
        if in_tf_function():
            return self._call_tf(inputs)
        outputs = self._call_ops(inputs)
        if in_grain_data_pipeline():
            # Grain pickles outputs across worker processes, so return NumPy
            # arrays rather than backend tensors.
            return convert_preprocessing_outputs_grain(outputs)
        return outputs

    def _normalize(self, video, convert):
        if self.scale is not None:
            video = video * convert(self.scale)
        if self.offset is not None:
            video = video + convert(self.offset)
        return video

    def _call_tf(self, inputs):
        video = tf.cast(inputs, "float32")
        frames = tf.shape(video)[0]
        target_h, target_w = _smart_resize_tf(
            frames,
            tf.shape(video)[1],
            tf.shape(video)[2],
            self.temporal_patch_size,
            self._factor,
            self.min_pixels,
            self.max_pixels,
        )
        if inputs.dtype.is_integer:
            video = resize_integer_pixels_tf(
                video, target_h, target_w, self.interpolation, self.antialias
            )
        else:
            video = tf.image.resize(
                video,
                (target_h, target_w),
                method=self.interpolation,
                antialias=self.antialias,
            )
        video = self._normalize(
            tf.clip_by_value(video, 0.0, 255.0),
            lambda v: tf.reshape(tf.constant(v, "float32"), (1, 1, 1, -1)),
        )
        t, p, m = (
            self.temporal_patch_size,
            self.patch_size,
            self.spatial_merge_size,
        )
        pad = -frames % t
        video = tf.concat([video, tf.tile(video[-1:], [pad, 1, 1, 1])], 0)
        grid_t = (frames + pad) // t
        grid_h = target_h // p
        grid_w = target_w // p
        video = tf.reshape(
            video, (grid_t, t, grid_h // m, m, p, grid_w // m, m, p, 3)
        )
        video = tf.transpose(video, (0, 2, 5, 3, 6, 1, 4, 7, 8))
        patches = tf.reshape(video, (grid_t * grid_h * grid_w, t, p, p, 3))
        return {
            "patches": patches,
            "grid_thw": tf.stack([grid_t, grid_h, grid_w]),
        }

    def _call_ops(self, inputs):
        inputs = ops.convert_to_tensor(inputs)
        is_int = is_int_dtype(inputs.dtype)
        video = ops.cast(inputs, "float32")
        frames, height, width = (int(d) for d in ops.shape(video)[:3])
        t, p, m = (
            self.temporal_patch_size,
            self.patch_size,
            self.spatial_merge_size,
        )
        target_h, target_w = smart_resize(
            frames,
            height,
            width,
            t,
            self._factor,
            self.min_pixels,
            self.max_pixels,
        )
        if is_int:
            video = resize_integer_pixels(
                video, target_h, target_w, self.interpolation, self.antialias
            )
        else:
            video = ops.image.resize(
                video,
                size=(target_h, target_w),
                interpolation=self.interpolation,
                antialias=self.antialias,
            )
        video = self._normalize(
            ops.clip(video, 0.0, 255.0),
            lambda v: ops.reshape(
                ops.convert_to_tensor(v, dtype="float32"), (1, 1, 1, -1)
            ),
        )
        pad = -frames % t
        if pad:
            video = ops.concatenate(
                [video, ops.tile(video[-1:], [pad, 1, 1, 1])], axis=0
            )
        grid_t = (frames + pad) // t
        grid_h = target_h // p
        grid_w = target_w // p
        video = ops.reshape(
            video, (grid_t, t, grid_h // m, m, p, grid_w // m, m, p, 3)
        )
        video = ops.transpose(video, (0, 2, 5, 3, 6, 1, 4, 7, 8))
        patches = ops.reshape(video, (grid_t * grid_h * grid_w, t, p, p, 3))
        grid_thw = ops.convert_to_tensor(
            [grid_t, grid_h, grid_w], dtype="int32"
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
                "fps": self.fps,
                "min_frames": self.min_frames,
                "max_frames": self.max_frames,
                "interpolation": self.interpolation,
                "antialias": self.antialias,
            }
        )
        return config
