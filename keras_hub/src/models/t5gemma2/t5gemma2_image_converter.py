from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.t5gemma2.t5gemma2_backbone import T5Gemma2Backbone


@keras_hub_export("keras_hub.layers.T5Gemma2ImageConverter")
class T5Gemma2ImageConverter(ImageConverter):
    backbone_cls = T5Gemma2Backbone

    def __init__(self, **kwargs):
        # Always do image preprocessing in float32
        kwargs.pop("dtype", None)
        dtype = "float32"
        super().__init__(dtype=dtype, **kwargs)
