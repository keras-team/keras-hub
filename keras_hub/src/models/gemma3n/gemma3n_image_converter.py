from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.gemma3n.gemma3n_backbone import Gemma3nBackbone


@keras_hub_export("keras_hub.layers.Gemma3nImageConverter")
class Gemma3nImageConverter(ImageConverter):
    backbone_cls = Gemma3nBackbone

    def __init__(self, **kwargs):
        # Always do image preprocessing in float32
        kwargs.pop("dtype", None)
        dtype = "float32"
        super().__init__(dtype=dtype, **kwargs)
