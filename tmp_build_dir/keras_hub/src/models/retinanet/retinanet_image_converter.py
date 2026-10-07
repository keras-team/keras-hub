from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.retinanet.retinanet_backbone import RetinaNetBackbone


@keras_hub_export("keras_hub.layers.RetinaNetImageConverter")
class RetinaNetImageConverter(ImageConverter):
    backbone_cls = RetinaNetBackbone
