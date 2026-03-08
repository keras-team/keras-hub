from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.image_classifier_preprocessor import (
    ImageClassifierPreprocessor,
)
from keras_hub.src.models.swin_transformer.swin_transformer_backbone import (
    SwinTransformerBackbone,
)
from keras_hub.src.models.swin_transformer.swin_transformer_image_converter import (  # noqa: E501
    SwinTransformerImageConverter,
)


@keras_hub_export("keras_hub.models.SwinTransformerImageClassifierPreprocessor")
class SwinTransformerImageClassifierPreprocessor(ImageClassifierPreprocessor):
    backbone_cls = SwinTransformerBackbone
    image_converter_cls = SwinTransformerImageConverter
