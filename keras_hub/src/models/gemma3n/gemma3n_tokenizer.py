from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.gemma3n.gemma3n_backbone import Gemma3nBackbone
from keras_hub.src.tokenizers.sentence_piece_tokenizer import (
    SentencePieceTokenizer,
)


@keras_hub_export(
    [
        "keras_hub.tokenizers.Gemma3nTokenizer",
        "keras_hub.models.Gemma3nTokenizer",
    ]
)
class Gemma3nTokenizer(SentencePieceTokenizer):
    """Gemma3n tokenizer layer based on SentencePiece.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_hub.tokenizers.SentencePieceTokenizer`. Unlike the
    underlying tokenizer, it will check for all special tokens needed by
    Gemma3n models and provides a `from_preset()` method to automatically
    download a matching vocabulary for a Gemma3n preset.

    If input is a batch of strings `(rank > 0)`, the layer will output a
    `tf.RaggedTensor` where the last dimension of the output is ragged.

    If input is a scalar string `(rank == 0)`, the layer will output a dense
    `tf.Tensor` with static shape `[None]`.

    Args:
        proto: Either a `string` path to a SentencePiece proto file, or a
            `bytes` object with a serialized SentencePiece proto. See the
            [SentencePiece repository](https://github.com/google/sentencepiece)
            for more details on the format.

    Examples:

    ```python
    # Unbatched input.
    tokenizer = keras_hub.models.Gemma3nTokenizer.from_preset(
        "gemma3n_instruct_1b"
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
    tokenizer = keras_hub.models.Gemma3nTokenizer(
        proto=bytes_io.getvalue(),
    )
    tokenizer("The quick brown fox jumped.")
    ```
    """

    backbone_cls = Gemma3nBackbone

    def __init__(self, proto, **kwargs):
        # Add special tokens.
        self._add_special_token("<bos>", "start_token")
        self._add_special_token("<eos>", "end_token")
        self._add_special_token("<pad>", "pad_token")
        # Image.
        self._add_special_token("<img>", "image_placeholder")
        # Audio.
        self._add_special_token("<audio_soft_token>", "audio_placeholder")
        # Multimodal inputs.
        self._add_special_token("<start_of_image>", "start_of_image_token")
        self._add_special_token("<end_of_image>", "end_of_image_token")
        self._add_special_token("<start_of_audio>", "start_of_audio_token")
        self._add_special_token("<end_of_audio>", "end_of_audio_token")
        # Special tokens for conversation and masking.
        self._add_special_token("<start_of_turn>", "start_of_turn_token")
        self._add_special_token("<end_of_turn>", "end_of_turn_token")
        self._add_special_token("<mask>", "mask_token")
        self._add_special_token("[multimodal]", "multimodal_token")
        super().__init__(proto=proto, **kwargs)
