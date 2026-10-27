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
        min_pixels: int. Minimum total pixels per frame after resizing.
        max_pixels: int. Maximum total pixels per frame after resizing.
        patch_size: int. Spatial patch size used by the vision encoder.
        temporal_patch_size: int. Number of frames grouped per temporal patch.
        merge_size: int. Spatial merge factor applied by the vision encoder.
        image_mean: tuple of float. Per-channel normalization mean.
        image_std: tuple of float. Per-channel normalization standard
            deviation.

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
        min_pixels=56 * 56,
        max_pixels=12845056,
        patch_size=14,
        temporal_patch_size=2,
        merge_size=2,
        image_mean=(0.48145466, 0.4578275, 0.40821073),
        image_std=(0.26862954, 0.26130258, 0.27577711),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.image_mean = np.array(image_mean, dtype="float32")
        self.image_std = np.array(image_std, dtype="float32")
        self.image_converter = Qwen2VLImageConverter(
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            merge_size=merge_size,
            image_mean=image_mean,
            image_std=image_std,
        )

    def call(self, video):
        return self.image_converter(video)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels,
                "patch_size": self.patch_size,
                "temporal_patch_size": self.temporal_patch_size,
                "merge_size": self.merge_size,
                "image_mean": self.image_mean.tolist(),
                "image_std": self.image_std.tolist(),
            }
        )
        return config
