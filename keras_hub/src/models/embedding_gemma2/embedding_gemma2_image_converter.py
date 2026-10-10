from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.embedding_gemma2.embedding_gemma2_backbone import (
    EmbeddingGemma2Backbone,
)
from keras_hub.src.models.gemma4.gemma4_image_converter import (
    Gemma4ImageConverter,
)


@keras_hub_export("keras_hub.models.EmbeddingGemma2ImageConverter")
class EmbeddingGemma2ImageConverter(Gemma4ImageConverter):
    """EmbeddingGemma2 image converter.

    This converter is a subclass of `Gemma4ImageConverter` and shares the same
    functionality.
    """

    backbone_cls = EmbeddingGemma2Backbone
