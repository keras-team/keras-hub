from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.gemma4.gemma4_backbone import Gemma4Backbone
from keras_hub.src.tokenizers.sentence_piece_tokenizer import (
    SentencePieceTokenizer,
)

# Gemma4 image token strings.
# Pattern: <|image> opens, <|image|> is the per-patch placeholder (repeated
# num_vision_tokens_per_image times), <image|> closes.
START_OF_IMAGE_TOKEN = "<|image>"
IMAGE_PLACEHOLDER_TOKEN = "<|image|>"
END_OF_IMAGE_TOKEN = "<image|>"

# Gemma4 audio token strings — mirror the image pattern.
START_OF_AUDIO_TOKEN = "<|audio>"
AUDIO_PLACEHOLDER_TOKEN = "<|audio|>"
END_OF_AUDIO_TOKEN = "<audio|>"


@keras_hub_export(
    [
        "keras_hub.tokenizers.Gemma4Tokenizer",
        "keras_hub.models.Gemma4Tokenizer",
    ]
)
class Gemma4Tokenizer(SentencePieceTokenizer):
    """Gemma4 tokenizer layer based on SentencePiece.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_hub.tokenizers.SentencePieceTokenizer`. Unlike the
    underlying tokenizer, it will check for all special tokens needed by
    Gemma4 models and provides a `from_preset()` method to automatically
    download a matching vocabulary for a Gemma4 preset.

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
    tokenizer = keras_hub.models.Gemma4Tokenizer.from_preset(
        "gemma4_instruct_4b"
    )
    tokenizer("The quick brown fox jumped.")

    # Batched input.
    tokenizer(["The quick brown fox jumped.", "The fox slept."])

    # Detokenization.
    tokenizer.detokenize(tokenizer("The quick brown fox jumped."))
    ```
    """

    backbone_cls = Gemma4Backbone

    def __init__(
        self, proto, has_vision_tokens=True, has_audio_tokens=False, **kwargs
    ):
        self.has_vision_tokens = has_vision_tokens
        self.has_audio_tokens = has_audio_tokens

        # Standard tokens.
        self._add_special_token("<bos>", "start_token")
        self._add_special_token("<eos>", "end_token")
        self._add_special_token("<pad>", "pad_token")

        if has_vision_tokens:
            # <|image>   — start of image region.
            # <|image|>  — per-patch placeholder (repeated per vision token).
            # <image|>   — end of image region.
            self._add_special_token(
                START_OF_IMAGE_TOKEN, "start_of_image_token"
            )
            self._add_special_token(
                IMAGE_PLACEHOLDER_TOKEN, "image_placeholder"
            )
            self._add_special_token(END_OF_IMAGE_TOKEN, "end_of_image_token")
        else:
            self.start_of_image_token_id = -1
            self.image_placeholder_id = -1
            self.end_of_image_token_id = -1

        if has_audio_tokens:
            # <|audio>   — start of audio region.
            # <|audio|>  — per-clip placeholder (repeated
            # num_audio_tokens_per_clip times).
            # <audio|>   — end of audio region.
            self._add_special_token(
                START_OF_AUDIO_TOKEN, "start_of_audio_token"
            )
            self._add_special_token(
                AUDIO_PLACEHOLDER_TOKEN, "audio_placeholder"
            )
            self._add_special_token(END_OF_AUDIO_TOKEN, "end_of_audio_token")
        else:
            self.start_of_audio_token_id = -1
            self.audio_placeholder_id = -1
            self.end_of_audio_token_id = -1

        super().__init__(proto=proto, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "has_vision_tokens": self.has_vision_tokens,
                "has_audio_tokens": self.has_audio_tokens,
            }
        )
        return config
