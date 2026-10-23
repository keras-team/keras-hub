from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.llama3.llama3_vision_backbone import (
    Llama3VisionBackbone,
)


@keras_hub_export("keras_hub.models.Llama3VisionImageConverter")
class Llama3VisionImageConverter(ImageConverter):
    """Image converter for the Llama 3.2 Vision model.

    This layer preprocesses images for the Llama 3.2 Vision model, handling
    resizing and rescaling to match the expected input format.

    Args:
        image_size: int or tuple. The target image size. Defaults to `None`.
        scale: float. The pixel value scale factor. Defaults to `1/255`.
        offset: float. The pixel value offset. Defaults to `0.0`.
        interpolation: str. The resize interpolation method.
            Defaults to `"bicubic"`.
        crop_to_aspect_ratio: bool. Whether to preserve aspect ratio.
            Defaults to `True`.
    """

    backbone_cls = Llama3VisionBackbone

    def __init__(
        self,
        image_size=None,
        scale=1.0 / 255.0,
        offset=0.0,
        interpolation="bicubic",
        crop_to_aspect_ratio=True,
        **kwargs,
    ):
        super().__init__(
            image_size=image_size,
            scale=scale,
            offset=offset,
            interpolation=interpolation,
            crop_to_aspect_ratio=crop_to_aspect_ratio,
            **kwargs,
        )
