from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.inception.inception_backbone import InceptionBackbone

@keras_hub_export("keras_hub.layers.InceptionImageConverter")
class InceptionImageConverter(ImageConverter):
    backbone_cls = InceptionBackbone