from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.image_classifier_preprocessor import (
    ImageClassifierPreprocessor,
)
from keras_hub.src.models.mix_transformer.mix_transformer_backbone import (
    MiTBackbone,
)
from keras_hub.src.models.mix_transformer.mix_transformer_image_converter import (
    MiTImageConverter,
)


@keras_hub_export("keras_hub.models.MiTImageClassifierPreprocessor")
class MiTImageClassifierPreprocessor(ImageClassifierPreprocessor):
    backbone_cls = MiTBackbone
    image_converter_cls = MiTImageConverter
