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

    Args:
        proto: Either a `string` path to a SentencePiece proto file, or a
            `bytes` object with a serialized SentencePiece proto. See the
            [SentencePiece repository](https://github.com/google/sentencepiece)
            for more details on the format.

    Examples:

    ```python
    # Unbatched input.
    tokenizer = keras_hub.models.EmbeddingGemma2Tokenizer.from_preset(
        "embedding_gemma2"
    )
    tokenizer("The quick brown fox jumped.")

    # Detokenization.
    tokenizer.detokenize(tokenizer("The quick brown fox jumped."))
    ```
    """

    backbone_cls = EmbeddingGemma2Backbone
