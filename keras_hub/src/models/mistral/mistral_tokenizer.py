from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.mistral.mistral_backbone import MistralBackbone
from keras_hub.src.tokenizers.sentence_piece_tokenizer import (
    SentencePieceTokenizer,
)


@keras_hub_export(
    [
        "keras_hub.tokenizers.MistralTokenizer",
        "keras_hub.models.MistralTokenizer",
    ]
)
class MistralTokenizer(SentencePieceTokenizer):
    """Mistral tokenizer layer based on SentencePiece.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_hub.tokenizers.SentencePieceTokenizer`. Unlike the
    underlying tokenizer, it will check for all special tokens needed by
    Mistral models and provides a `from_preset()` method to automatically
    download a matching vocabulary for a Mistral preset.

    If input is a batch of strings (rank > 0), the layer will output a
    `tf.RaggedTensor` where the last dimension of the output is ragged.

    If input is a scalar string (rank == 0), the layer will output a dense
    `tf.Tensor` with static shape `[None]`.

    Args:
        proto: Either a `string` path to a SentencePiece proto file, or a
            `bytes` object with a serialized SentencePiece proto. See the
            [SentencePiece repository](https://github.com/google/sentencepiece)
            for more details on the format.

    Examples:
    ```python
    # Unbatched input.
    tokenizer = keras_hub.models.MistralTokenizer.from_preset(
        "mistral_7b_en",
    )
    tokenizer("The quick brown fox jumped.")

    # Batched input.
    tokenizer(["The quick brown fox jumped.", "The fox slept."])

    # Detokenization.
    tokenizer.detokenize(tokenizer("The quick brown fox jumped."))
    ```
    """

    backbone_cls = MistralBackbone

    def __init__(self, proto, **kwargs):
        self._add_special_token("<s>", "start_token")
        self._add_special_token("</s>", "end_token")
        self.pad_token_id = 0
        super().__init__(proto=proto, **kwargs)
