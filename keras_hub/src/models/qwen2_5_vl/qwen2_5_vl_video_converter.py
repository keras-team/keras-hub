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

    Same frame preprocessing as Qwen2-VL — this subclass only defaults to
    a ``Qwen2_5VLImageConverter`` for preset resolution. Video fps
    metadata does not flow through converters; pass per-video
    ``second_per_grid_ts`` in preprocessor inputs instead.

    Args:
        image_converter: ``Qwen2_5VLImageConverter`` instance, or ``None``
            to build one with default configuration.
    """

    backbone_cls = Qwen2_5VLBackbone

    def __init__(self, image_converter=None, **kwargs):
        super().__init__(
            image_converter=image_converter or Qwen2_5VLImageConverter(),
            **kwargs,
        )
