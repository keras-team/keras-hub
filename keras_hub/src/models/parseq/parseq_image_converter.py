from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.parseq.parseq_backbone import PARSeqBackbone


@keras_hub_export("keras_hub.layers.PARSeqImageConverter")
class PARSeqImageConverter(ImageConverter):
    backbone_cls = PARSeqBackbone
