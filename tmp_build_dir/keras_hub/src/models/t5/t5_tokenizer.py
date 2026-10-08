from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.t5.t5_backbone import T5Backbone
from keras_hub.src.tokenizers.sentence_piece_tokenizer import (
    SentencePieceTokenizer,
)


@keras_hub_export(
    [
        "keras_hub.tokenizers.T5Tokenizer",
        "keras_hub.models.T5Tokenizer",
    ]
)
class T5Tokenizer(SentencePieceTokenizer):
    """T5 tokenizer layer based on SentencePiece.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_hub.tokenizers.SentencePieceTokenizer`. Unlike the
    underlying tokenizer, it will check for all special tokens needed by
    T5 models and provides a `from_preset()` method to automatically
    download a matching vocabulary for a T5 preset.

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
    bytes_io = io.BytesIO()
    ds = tf.data.Dataset.from_tensor_slices(["The quick brown fox jumped."])
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=ds.as_numpy_iterator(),
        model_writer=bytes_io,
        vocab_size=8,
        model_type="WORD",
        bos_id=-1,
        pad_id=0,
        eos_id=1,
        unk_id=2,
        pad_piece="<pad>",
        eos_piece="</s>",
        unk_piece="<unk>",
    )
    tokenizer = keras_hub.models.T5Tokenizer(
        proto=bytes_io.getvalue(),
    )
    tokenizer("The quick brown fox jumped.")

    # Batched inputs.
    tokenizer(["the quick brown fox", "the earth is round"])

    # Unbatched inputs.
    tokenizer("the quick brown fox")

    # Detokenization.
    tokenizer.detokenize(tokenizer("The quick brown fox jumped."))
    ```
    """

    backbone_cls = T5Backbone

    def __init__(self, proto, **kwargs):
        # T5 uses the same start token as end token, i.e., "<\s>".
        self._add_special_token("</s>", "end_token")
        self._add_special_token("</s>", "start_token")
        self._add_special_token("<pad>", "pad_token")
        super().__init__(proto=proto, **kwargs)
