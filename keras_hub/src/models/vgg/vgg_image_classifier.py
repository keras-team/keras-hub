from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.image_classifier import ImageClassifier
from keras_hub.src.models.vgg.vgg_backbone import VGGBackbone


@keras_hub_export("keras_hub.models.VGGImageClassifier")
class VGGImageClassifier(ImageClassifier):
    backbone_cls = VGGBackbone
