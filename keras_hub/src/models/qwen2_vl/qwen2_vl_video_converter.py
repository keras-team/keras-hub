import keras

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
            standard Qwen2-VL configuration.

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

    def __init__(self, image_converter=None, **kwargs):
        super().__init__(**kwargs)
        self.image_converter = image_converter or Qwen2VLImageConverter()

    @property
    def merge_size(self):
        return self.image_converter.merge_size

    def call(self, video):
        return self.image_converter(video)

    def get_config(self):
        config = super().get_config()
        config["image_converter"] = keras.layers.serialize(self.image_converter)
        return config

    @classmethod
    def from_config(cls, config):
        config["image_converter"] = keras.layers.deserialize(
            config["image_converter"]
        )
        return cls(**config)
