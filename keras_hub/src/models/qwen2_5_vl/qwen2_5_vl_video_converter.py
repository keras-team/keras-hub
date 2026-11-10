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

    Same frame preprocessing as Qwen2-VL — this subclass defaults to
    a ``Qwen2_5VLImageConverter`` for preset resolution. Unlike Qwen2-VL
    (whose checkpoints ship a separate ``video_preprocessor_config.json``),
    Qwen2.5-VL inherits its per-frame pixel bounds from the image
    processor, so ``min_pixels``/``max_pixels`` default to the wrapped
    image converter's bounds. Video fps metadata does not flow through
    converters; pass per-video ``second_per_grid_ts`` in preprocessor
    inputs instead.

    Args:
        image_converter: ``Qwen2_5VLImageConverter`` instance, or ``None``
            to build one with default configuration.
        min_pixels: int or ``None``. Minimum total pixel count per frame
            after resize. ``None`` inherits ``image_converter.min_pixels``
            (``3136`` for a default converter), matching Hugging Face's
            image-processor fallback for Qwen2.5-VL videos.
        max_pixels: int or ``None``. Maximum total pixel count per frame
            after resize. ``None`` inherits ``image_converter.max_pixels``
            (``12845056`` for a default converter), matching Hugging Face's
            image-processor fallback for Qwen2.5-VL videos.
    """

    backbone_cls = Qwen2_5VLBackbone

    def __init__(
        self, image_converter=None, min_pixels=None, max_pixels=None, **kwargs
    ):
        image_converter = image_converter or Qwen2_5VLImageConverter()
        super().__init__(
            image_converter=image_converter,
            min_pixels=(
                image_converter.min_pixels if min_pixels is None else min_pixels
            ),
            max_pixels=(
                image_converter.max_pixels if max_pixels is None else max_pixels
            ),
            **kwargs,
        )
