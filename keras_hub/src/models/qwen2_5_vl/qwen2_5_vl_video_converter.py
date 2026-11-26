from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.qwen2_5_vl.qwen2_5_vl_backbone import (
    Qwen2_5VLBackbone,
)
from keras_hub.src.models.qwen2_5_vl.qwen2_5_vl_image_converter import (
    Qwen2_5VLImageConverter,
)
from keras_hub.src.models.qwen2_vl.qwen2_vl_video_converter import (
    Qwen2VLVideoConverter,
)


@keras_hub_export("keras_hub.layers.Qwen2_5VLVideoConverter")
class Qwen2_5VLVideoConverter(Qwen2VLVideoConverter):
    """Qwen2.5-VL video converter.

    Frame preprocessing matches Qwen2-VL. The only difference is the
    default nested converter: when ``image_converter`` is ``None``, a
    ``Qwen2_5VLImageConverter`` is built. Temporal scaling is not part of
    conversion; pass per-video ``second_per_grid_ts`` in preprocessor
    inputs instead.

    Args:
        image_converter: ``Qwen2_5VLImageConverter`` instance, or ``None``
            to build one with default configuration. Defaults to ``None``.
        min_pixels: int or ``None``. Per-frame pixel floor; ``None``
            inherits from ``image_converter``. Defaults to ``None``.
        max_pixels: int or ``None``. Per-frame pixel cap; ``None``
            inherits from ``image_converter``. Defaults to ``None``.
        cap_pixels_per_frame: bool. Whether to bound each frame's resized
            pixel count so the whole video stays within an even share of
            the ``max_video_tokens`` budget (90% of it), matching the
            opt-in ``cap_pixels_per_frame`` flag in Hugging Face's
            ``Qwen2VLVideoProcessor``. Defaults to ``False``.
        max_video_tokens: int. Video token budget used when
            ``cap_pixels_per_frame`` is enabled. Defaults to ``128000``.

    Examples:

    ```python
    # Single clip: (num_frames, height, width, channels).
    converter = keras_hub.layers.Qwen2_5VLVideoConverter()
    converter(np.random.randint(0, 256, size=(8, 224, 224, 3)))

    # Multiple clips.
    converter(
        [np.random.randint(0, 256, size=(8, 224, 224, 3)),
         np.random.randint(0, 256, size=(4, 224, 224, 3))]
    )
    ```
    """

    backbone_cls = Qwen2_5VLBackbone

    def __init__(self, image_converter=None, **kwargs):
        super().__init__(
            image_converter=image_converter or Qwen2_5VLImageConverter(),
            **kwargs,
        )
