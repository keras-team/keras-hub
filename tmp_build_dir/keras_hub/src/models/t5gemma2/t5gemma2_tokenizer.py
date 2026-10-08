from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.t5gemma2.t5gemma2_backbone import T5Gemma2Backbone
from keras_hub.src.tokenizers.sentence_piece_tokenizer import (
    SentencePieceTokenizer,
)


@keras_hub_export(
    [
        "keras_hub.tokenizers.T5Gemma2Tokenizer",
        "keras_hub.models.T5Gemma2Tokenizer",
    ]
)
class T5Gemma2Tokenizer(SentencePieceTokenizer):
    """T5Gemma2 tokenizer layer based on SentencePiece.

    This tokenizer class will tokenize raw strings into integer sequences
    and is based on `keras_hub.tokenizers.SentencePieceTokenizer`. Unlike
    the underlying tokenizer, it will check for all special tokens needed
    by T5Gemma2 models and provides a `from_preset()` method to
    automatically download a matching vocabulary for a T5Gemma2 preset.

    If input is a batch of strings (rank > 0), the layer will output a
    `tf.RaggedTensor` where the last dimension of the output is ragged.

    If input is a scalar string (rank == 0), the layer will output a
    dense `tf.Tensor` with static shape `[None]`.

    Args:
        proto: Either a `string` path to a SentencePiece proto file,
            or a `bytes` object with a serialized SentencePiece proto.

    Examples:

    ```python
    tokenizer = keras_hub.models.T5Gemma2Tokenizer.from_preset(
        "t5gemma2_270m_270m"
    )
    tokenizer("The quick brown fox jumped.")

    # Batched input.
    tokenizer(["The quick brown fox jumped.", "The fox slept."])

    # Detokenization.
    tokenizer.detokenize(tokenizer("The quick brown fox jumped."))
    ```
    """

    backbone_cls = T5Gemma2Backbone

    def __init__(self, proto, **kwargs):
        self._add_special_token("<bos>", "start_token")
        self._add_special_token("<eos>", "end_token")
        self._add_special_token("<pad>", "pad_token")
        super().__init__(proto=proto, **kwargs)
