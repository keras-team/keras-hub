"""MetaCLIP 2 image converter implementation."""

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.image_converter import ImageConverter
from keras_hub.src.models.metaclip_2.metaclip_2_backbone import (
    MetaCLIP2Backbone,
)


@keras_hub_export("keras_hub.layers.MetaCLIP2ImageConverter")
class MetaCLIP2ImageConverter(ImageConverter):
    """Image converter for MetaCLIP 2 models.

    This converter handles image preprocessing for MetaCLIP 2, including
    resizing and normalization to match the model's expected input format.
    """

    backbone_cls = MetaCLIP2Backbone
