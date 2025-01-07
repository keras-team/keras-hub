from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.parseq.parseq_backbone import ParseQBackbone


@keras_hub_export("keras_hub.layers.ParseQImageConverter")
class DiffBinImageConverter(ImageConverter):
    backbone_cls = ParseQBackbone
