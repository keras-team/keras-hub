import re

from keras_hub.src.tests.mocks.mock_gemma_tokenizer_base import (
    MockGemmaTokenizerBase,
)
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import tf


class MockGemma4Tokenizer(MockGemmaTokenizerBase):
    _strip_text = True

    def _setup_tokens(self):
        self.vocabulary = [
            "<pad>",
            "<bos>",
            "<eos>",
            "<unk>",
            "<|image>",
            "<image|>",
            "<start_of_turn>",
            "<end_of_turn>",
            "<|image|>",
            "the",
            "brown",
            "earth",
            "fox",
            "is",
            "quick",
            "round",
            "\n\n",
            "<|turn>",
            "<turn|>",
            "<|audio|>",
            "<|audio>",
            "<audio|>",
            "<|video|>",
            "<|video>",
            "<video|>",
        ]

        # Image placeholder token.
        self._add_special_token("<|image|>", "image_placeholder")

        # Boundary tokens used in the preprocessor.
        self._add_special_token("<|image>", "start_of_image_token")
        self._add_special_token("<image|>", "end_of_image_token")

        # Video tokens.
        self._add_special_token("<|video|>", "video_placeholder")
        self._add_special_token("<|video>", "start_of_video_token")
        self._add_special_token("<video|>", "end_of_video_token")

        self.audio_placeholder_id = -1

        self._space_padded_tokens = (
            self.start_of_image_token,
            self.end_of_image_token,
            self.image_placeholder,
            "<|audio>",
            "<audio|>",
            "<|audio|>",
            self.start_of_video_token,
            self.end_of_video_token,
            self.video_placeholder,
        )

    @preprocessing_function
    def _tokenize_tf(self, inputs):
        self._maybe_initialized_tf()
        inputs = tf.convert_to_tensor(inputs)
        unbatched = inputs.shape.rank == 0
        if unbatched:
            inputs = tf.expand_dims(inputs, 0)

        inputs = tf.strings.regex_replace(
            inputs,
            re.escape(self.start_of_image_token),
            f" {self.start_of_image_token} ",
        )
        inputs = tf.strings.regex_replace(
            inputs,
            re.escape(self.end_of_image_token),
            f" {self.end_of_image_token} ",
        )
        inputs = tf.strings.regex_replace(
            inputs,
            re.escape(self.image_placeholder),
            f" {self.image_placeholder} ",
        )
        inputs = tf.strings.regex_replace(
            inputs,
            re.escape("<|audio>"),
            " <|audio> ",
        )
        inputs = tf.strings.regex_replace(
            inputs,
            re.escape("<audio|>"),
            " <audio|> ",
        )
        inputs = tf.strings.regex_replace(
            inputs,
            re.escape("<|audio|>"),
            " <|audio|> ",
        )
        inputs = tf.strings.regex_replace(
            inputs,
            re.escape("<|video>"),
            " <|video> ",
        )
        inputs = tf.strings.regex_replace(
            inputs,
            re.escape("<video|>"),
            " <video|> ",
        )
        inputs = tf.strings.regex_replace(
            inputs,
            re.escape("<|video|>"),
            " <|video|> ",
        )
        inputs = tf.strings.regex_replace(inputs, "  ", " ")
        inputs = tf.strings.strip(inputs)

        sep_inputs = tf.strings.split(inputs, sep=" ")
        tokens = self.string_to_id.lookup(sep_inputs)

        if self.add_bos:
            bos_tensor = tf.fill(
                value=self.start_token_id,
                dims=tokens.shape.as_list()[0:1] + [1],
            )
            tokens = tf.concat((bos_tensor, tokens), axis=-1)
        if self.add_eos:
            eos_tensor = tf.fill(
                value=self.end_token_id,
                dims=tokens.shape.as_list()[0:1] + [1],
            )
            tokens = tf.concat((tokens, eos_tensor), axis=-1)

        if unbatched:
            tokens = tf.squeeze(tokens, 0)

        return self._cast_tokens_tf(tokens)
