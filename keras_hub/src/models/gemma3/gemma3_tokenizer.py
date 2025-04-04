from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.gemma3.gemma3_backbone import Gemma3Backbone
from keras_hub.src.tokenizers.sentence_piece_tokenizer import (
    SentencePieceTokenizer,
)

START_OF_IMAGE_TOKEN = "<start_of_image>"
IMAGE_PLACEHOLDER_TOKEN = "<img>"
END_OF_IMAGE_TOKEN = "<end_of_image>"


@keras_hub_export(
    [
        "keras_hub.tokenizers.Gemma3Tokenizer",
        "keras_hub.models.Gemma3Tokenizer",
    ]
)
class Gemma3Tokenizer(SentencePieceTokenizer):
    """Gemma tokenizer layer based on SentencePiece.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_hub.tokenizers.SentencePieceTokenizer`. Unlike the
    underlying tokenizer, it will check for all special tokens needed by
    Gemma models and provides a `from_preset()` method to automatically
    download a matching vocabulary for a Gemma preset.

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
    tokenizer = keras_hub.models.Gemma3Tokenizer.from_preset(
        "gemma_instruct_1b"
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
    tokenizer = keras_hub.models.Gemma3Tokenizer(
        proto=bytes_io.getvalue(),
    )
    tokenizer("The quick brown fox jumped.")
    ```
    """

    backbone_cls = Gemma3Backbone

    def __init__(self, proto, **kwargs):
        # Add special tokens.

        # The usual tokens.
        self._add_special_token("<bos>", "start_token")
        self._add_special_token("<eos>", "end_token")
        self._add_special_token("<pad>", "pad_token")

        # Image placeholder token.
        self._add_special_token("<img>", "image_placeholder")

        #  Some tokens which are used in the preprocessor. We need to keep them
        # here so that the preprocessor works with `tf.data`.
        self._add_special_token("<start_of_image>", "start_of_image_token")
        self._add_special_token("<end_of_image>", "end_of_image_token")

        super().__init__(proto=proto, **kwargs)
