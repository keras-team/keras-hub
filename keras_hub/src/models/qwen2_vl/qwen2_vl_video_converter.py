import keras
import numpy as np

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.video_converter import VideoConverter
from keras_hub.src.models.qwen2_vl.qwen2_vl_backbone import Qwen2VLBackbone
from keras_hub.src.models.qwen2_vl.qwen2_vl_image_converter import (
    Qwen2VLImageConverter,
)


@keras_hub_export("keras_hub.layers.Qwen2VLVideoConverter")
class Qwen2VLVideoConverter(VideoConverter):
    """Qwen2-VL video converter.

    Converts video clips into the patch feature format consumed by
    `Qwen2VLBackbone`. Qwen2-VL applies identical spatial resizing,
    normalization, and patch merging to video frames as it does to images, so
    this layer delegates to `Qwen2VLImageConverter` after treating the video
    as a stack of frames.

    Args:
        image_converter: `Qwen2VLImageConverter`. The converter applied to
            each frame stack. Pass an instance to customize resizing or
            normalization; defaults to a `Qwen2VLImageConverter` with the
            standard Qwen2-VL configuration. Per-frame pixel bounds also
            default to this converter's bounds, matching Hugging Face
            where ``from_pretrained`` maps the image processor's
            ``min_pixels``/``max_pixels`` onto the video processor's size.
        min_pixels: int or ``None``. Minimum total pixel count per frame
            after resize. ``None`` inherits ``image_converter.min_pixels``
            (``3136`` for a default converter).
        max_pixels: int or ``None``. Maximum total pixel count per frame
            after resize. ``None`` inherits ``image_converter.max_pixels``
            (``12845056`` for a default converter).
        cap_pixels_per_frame: bool. Whether to bound each frame's resized
            pixel count so the whole video stays within an even share of
            the ``max_video_tokens`` budget (90% of it), matching the
            opt-in ``cap_pixels_per_frame`` flag in Hugging Face's
            ``Qwen2VLVideoProcessor``. Defaults to ``False`` — the same as
            current Hugging Face behavior.
        max_video_tokens: int. Video token budget used when
            ``cap_pixels_per_frame`` is enabled. Defaults to ``128000``,
            matching Hugging Face.

    Call arguments:
        video: NumPy array of shape ``(T, H, W, C)`` for a single video clip,
            or a **list** of NumPy arrays (each ``(T, H, W, C)``) for multiple
            video clips. Pixel values should be in ``[0, 255]``.

    Returns:
        Dict with keys:
        - ``"patches"``: float32 NumPy array of shape
          ``(total_patches, temporal_patch_size, patch_size, patch_size, C)``.
        - ``"grid_thw"``: int32 NumPy array of shape ``(num_videos, 3)`` with
          ``[grid_t, grid_h, grid_w]`` per video.
    """

    backbone_cls = Qwen2VLBackbone

    def __init__(
        self,
        image_converter=None,
        min_pixels=None,
        max_pixels=None,
        cap_pixels_per_frame=False,
        max_video_tokens=128000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_converter = image_converter or Qwen2VLImageConverter()
        self.min_pixels = (
            self.image_converter.min_pixels
            if min_pixels is None
            else min_pixels
        )
        self.max_pixels = (
            self.image_converter.max_pixels
            if max_pixels is None
            else max_pixels
        )
        self.cap_pixels_per_frame = cap_pixels_per_frame
        self.max_video_tokens = max_video_tokens

    @property
    def merge_size(self):
        return self.image_converter.merge_size

    def call(self, video):
        # The per-frame budget depends on each video's own frame count,
        # so lists are processed clip-by-clip and concatenated.
        if isinstance(video, list):
            all_patches, all_grids = [], []
            for clip in video:
                out = self._convert_clip(clip)
                all_patches.append(out["patches"])
                all_grids.append(out["grid_thw"])
            return {
                "patches": np.concatenate(all_patches, axis=0),
                "grid_thw": np.concatenate(all_grids, axis=0),
            }
        return self._convert_clip(video)

    def _convert_clip(self, video):
        image_converter = self.image_converter
        max_pixels = self.max_pixels
        if self.cap_pixels_per_frame:
            factor = image_converter.patch_size * image_converter.merge_size
            total_pixels = int(self.max_video_tokens * factor * factor * 0.9)
            num_frames = video.shape[0]
            # Even share of the token budget per frame, floored at
            # ~min_pixels so very long clips still produce usable frames
            # (matches the Hugging Face formula).
            max_pixels = max(
                min(
                    max_pixels,
                    total_pixels
                    * image_converter.temporal_patch_size
                    // num_frames,
                ),
                int(self.min_pixels * 1.05),
            )
        return image_converter(
            video, min_pixels=self.min_pixels, max_pixels=max_pixels
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "image_converter": keras.layers.serialize(self.image_converter),
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels,
                "cap_pixels_per_frame": self.cap_pixels_per_frame,
                "max_video_tokens": self.max_video_tokens,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["image_converter"] = keras.layers.deserialize(
            config["image_converter"]
        )
        return cls(**config)
