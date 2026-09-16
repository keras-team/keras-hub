"""Tokenizer for TIPSv2 models."""

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.tipsv2.tipsv2_backbone import TIPSv2Backbone
from keras_hub.src.tokenizers.sentence_piece_tokenizer import (
    SentencePieceTokenizer,
)


@keras_hub_export(
    [
        "keras_hub.tokenizers.TIPSv2Tokenizer",
        "keras_hub.models.TIPSv2Tokenizer",
    ]
)
class TIPSv2Tokenizer(SentencePieceTokenizer):
    """TIPSv2 tokenizer layer based on SentencePiece.

    This tokenizer wraps a SentencePiece model for TIPSv2 text encoding.
    Unlike CLIP/SigLIP, TIPSv2 does not add BOS/EOS tokens and expects
    lowercased input text.

    If input is a batch of strings (rank > 0), the layer will output a
    `tf.RaggedTensor` where the last dimension is ragged.

    If input is a scalar string (rank == 0), the layer will output a dense
    `tf.Tensor` with static shape `[None]`.

    Args:
        proto: Either a `string` path to a SentencePiece proto file, or a
            `bytes` object with a serialized SentencePiece proto.

    Examples:
    ```python
    tokenizer = keras_hub.models.TIPSv2Tokenizer.from_preset(
        "tipsv2_b14"
    )
    tokenizer("a photo of a cat")
    ```
    """

    backbone_cls = TIPSv2Backbone

    def __init__(self, proto, **kwargs):
        self._add_special_token("<unk>", "unknown_token")
        super().__init__(proto=proto, **kwargs)
