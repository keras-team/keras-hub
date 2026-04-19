from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.moondream.moondream_backbone import MoondreamBackbone
from keras_hub.src.tokenizers.sentence_piece_tokenizer import (
    SentencePieceTokenizer,
)


@keras_hub_export(
    [
        "keras_hub.tokenizers.MoondreamTokenizer",
        "keras_hub.models.MoondreamTokenizer",
    ]
)
class MoondreamTokenizer(SentencePieceTokenizer):
    """Moondream tokenizer layer based on SentencePiece.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_hub.tokenizers.SentencePieceTokenizer`. Unlike the
    underlying tokenizer, it will check for all special tokens needed by
    Moondream models and provides a `from_preset()` method to automatically
    download a matching vocabulary for a Moondream preset.

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
    tokenizer = keras_hub.models.MoondreamTokenizer.from_preset(
        "moondream2"
    )
    tokenizer("The quick brown fox jumped.")

    # Batched input.
    tokenizer(["The quick brown fox jumped.", "The fox slept."])

    # Detokenization.
    tokenizer.detokenize(tokenizer("The quick brown fox jumped."))
    ```
    """

    backbone_cls = MoondreamBackbone

    def __init__(self, proto, **kwargs):
        super().__init__(proto=proto, **kwargs)

    @property
    def start_token_id(self):
        return self.token_to_id("<s>")

    @property
    def end_token_id(self):
        return self.token_to_id("</s>")

    @property
    def pad_token_id(self):
        return self.token_to_id("<unk>")
