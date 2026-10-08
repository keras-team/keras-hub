from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.sam3.sam3_pc_backbone import (
    SAM3PromptableConceptBackbone,
)


@keras_hub_export("keras_hub.layers.SAM3ImageConverter")
class SAM3ImageConverter(ImageConverter):
    backbone_cls = SAM3PromptableConceptBackbone
