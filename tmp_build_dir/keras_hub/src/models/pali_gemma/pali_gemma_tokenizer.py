from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.gemma.gemma_tokenizer import GemmaTokenizer
from keras_hub.src.models.pali_gemma.pali_gemma_backbone import (
    PaliGemmaBackbone,
)


@keras_hub_export(
    [
        "keras_hub.tokenizers.PaliGemmaTokenizer",
        "keras_hub.models.PaliGemmaTokenizer",
    ]
)
class PaliGemmaTokenizer(GemmaTokenizer):
    """PaliGemma tokenizer layer based on SentencePiece.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_hub.tokenizers.SentencePieceTokenizer`. Unlike the
    underlying tokenizer, it will check for all special tokens needed by
    PaliGemma models and provides a `from_preset()` method to automatically
    download a matching vocabulary for a PaliGemma preset.

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
    tokenizer = keras_hub.models.PaliGemmaTokenizer.from_preset(
        "pali_gemma_3b_224"
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
        pad_piece="<pad>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        unk_piece="<unk>",
    )
    tokenizer = keras_hub.models.PaliGemmaTokenizer(
        proto=bytes_io.getvalue(),
    )
    tokenizer("The quick brown fox jumped.")
    ```
    """

    backbone_cls = PaliGemmaBackbone
