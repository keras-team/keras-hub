"""Qwen2-VL Image Converter.

Preprocesses raw images into model-ready inputs (resize, rescale,
normalize). The Qwen2-VL vision encoder expects float32 images.
"""

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.qwen2_vl.qwen2_vl_backbone import Qwen2VLBackbone


@keras_hub_export("keras_hub.layers.Qwen2VLImageConverter")
class Qwen2VLImageConverter(ImageConverter):
    """Image converter for Qwen2-VL models.

    This layer handles image preprocessing (resize, normalize) for the
    Qwen2-VL vision encoder.  Image processing is always performed in
    ``float32``.
    """

    backbone_cls = Qwen2VLBackbone

    def __init__(self, **kwargs):
        # Always do image preprocessing in float32.
        kwargs.pop("dtype", None)
        dtype = "float32"
        super().__init__(dtype=dtype, **kwargs)
