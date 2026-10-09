from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.embedding_gemma2.embedding_gemma2_backbone import (
    EmbeddingGemma2Backbone,
)
from keras_hub.src.models.gemma4.gemma4_tokenizer import Gemma4Tokenizer


@keras_hub_export("keras_hub.models.EmbeddingGemma2Tokenizer")
class EmbeddingGemma2Tokenizer(Gemma4Tokenizer):
    """EmbeddingGemma2 tokenizer.

    This tokenizer is a subclass of `Gemma4Tokenizer` and shares the same
    vocabulary and functionality.
    """

    backbone_cls = EmbeddingGemma2Backbone
