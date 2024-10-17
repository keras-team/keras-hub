from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.segformer.segformer_backbone import SegFormerBackbone


@keras_hub_export("keras_hub.layers.SegFormerImageConverter")
class SegFormerImageConverter(ImageConverter):
    backbone_cls = SegFormerBackbone
