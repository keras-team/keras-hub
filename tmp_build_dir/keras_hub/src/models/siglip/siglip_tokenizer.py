from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.siglip.siglip_backbone import SigLIPBackbone
from keras_hub.src.tokenizers.sentence_piece_tokenizer import (
    SentencePieceTokenizer,
)


@keras_hub_export(
    [
        "keras_hub.tokenizers.SigLIPTokenizer",
        "keras_hub.models.SigLIPTokenizer",
    ]
)
class SigLIPTokenizer(SentencePieceTokenizer):
    """SigLIP tokenizer layer based on SentencePiece.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_hub.tokenizers.SentencePieceTokenizer`. Unlike the
    underlying tokenizer, it will check for all special tokens needed by
    SigLIP models and provides a `from_preset()` method to automatically
    download a matching vocabulary for a SigLIP preset.

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
    tokenizer = keras_hub.models.SigLIPTokenizer.from_preset(
        "siglip_base_patch16_224"
    )
    tokenizer("The quick brown fox jumped.")

    # Batched input.
    tokenizer(["The quick brown fox jumped.", "The fox slept."])

    # Detokenization.
    tokenizer.detokenize(tokenizer("The quick brown fox jumped."))

    # Custom vocabulary.
    bytes_io = io.BytesIO()
    ds = tf.data.Dataset.from_tensor_slices(["The quick brown fox jumped."])
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=ds.as_numpy_iterator(),
        model_writer=bytes_io,
        vocab_size=8,
        model_type="WORD",
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        unk_piece="<unk>",
    )
    tokenizer = keras_hub.models.SigLIPTokenizer(
        proto=bytes_io.getvalue(),
    )
    tokenizer("The quick brown fox jumped.")
    ```
    """

    backbone_cls = SigLIPBackbone

    def __init__(self, proto, **kwargs):
        self._add_special_token("<unk>", "unknown_token")
        self._add_special_token("</s>", "end_token")
        self._add_special_token("</s>", "pad_token")
        super().__init__(proto=proto, **kwargs)
