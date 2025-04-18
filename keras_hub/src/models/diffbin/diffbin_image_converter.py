from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.diffbin.diffbin_backbone import DiffBinBackbone


@keras_hub_export("keras_hub.layers.DiffBinImageConverter")
class DiffBinImageConverter(ImageConverter):
    backbone_cls = DiffBinBackbone
