from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.video_prism.video_prism_backbone import (
    VideoPrismBackbone,
)
from keras_hub.src.tokenizers.sentence_piece_tokenizer import (
    SentencePieceTokenizer,
)


@keras_hub_export(
    [
        "keras_hub.tokenizers.VideoPrismTokenizer",
        "keras_hub.models.VideoPrismTokenizer",
    ]
)
class VideoPrismTokenizer(SentencePieceTokenizer):
    """VideoPrism tokenizer layer based on SentencePiece.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_hub.tokenizers.SentencePieceTokenizer`.

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
    tokenizer = keras_hub.models.VideoPrismTokenizer.from_preset(
        "videoprism_lvt_public_v1_base"
    )
    tokenizer("The quick brown fox jumped.")

    # Batched input.
    tokenizer(["The quick brown fox jumped.", "The fox slept."])

    # Detokenization.
    tokenizer.detokenize(tokenizer("The quick brown fox jumped."))
    ```
    """

    backbone_cls = VideoPrismBackbone

    def __init__(self, proto, **kwargs):
        super().__init__(proto=proto, **kwargs)
