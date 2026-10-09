from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.embedding_gemma2.embedding_gemma2_backbone import (
    EmbeddingGemma2Backbone,
)
from keras_hub.src.models.gemma4.gemma4_audio_converter import (
    Gemma4AudioConverter,
)


@keras_hub_export("keras_hub.models.EmbeddingGemma2AudioConverter")
class EmbeddingGemma2AudioConverter(Gemma4AudioConverter):
    """EmbeddingGemma2 audio converter.

    This converter is a subclass of `Gemma4AudioConverter` and shares the same
    functionality.
    """

    backbone_cls = EmbeddingGemma2Backbone
