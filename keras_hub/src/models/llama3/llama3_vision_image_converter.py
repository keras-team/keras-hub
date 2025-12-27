from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.llama3.llama3_vision_backbone import (
    Llama3VisionBackbone,
)


@keras_hub_export("keras_hub.models.Llama3VisionImageConverter")
class Llama3VisionImageConverter(ImageConverter):
    """Llama 3 Vision Image Converter.

    This layer preprocesses image inputs for the Llama 3 Vision model.
    It handles resizing and rescaling of images to match the model's
    expected input format (SigLIP style).

    Args:
        image_size: int or tuple (height, width). The target size for
            the images. If None, defaults to the config of the associated
            model.
        scale: float. The scale factor for pixel values.
            Defaults to 1/255 (rescaling [0, 255] to [0, 1]).
        offset: float. The offset for pixel values. Defaults to 0.0.
        interpolation: str. The interpolation method for resizing.
            Defaults to "bicubic".
        crop_to_aspect_ratio: bool. Whether to resize images while preserving
            aspect ratio. Defaults to True.
        **kwargs: Arguments passed to the parent `ImageConverter`.
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
