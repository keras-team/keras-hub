from keras_hub.src.tests.mocks.mock_gemma_tokenizer_base import (
    MockGemmaTokenizerBase,
)
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import tf


class MockGemma3Tokenizer(MockGemmaTokenizerBase):
    def _setup_tokens(self):
        self.vocabulary = [
            "<pad>",
            "<bos>",
            "<eos>",
            "<unk>",
            "<start_of_image>",
            "<end_of_image>",
            "<start_of_turn>",
            "<end_of_turn>",
            "<img>",
            "the",
            "brown",
            "earth",
            "fox",
            "is",
            "quick",
            "round",
            "\n\n",
        ]

        # Image placeholder token.
        self._add_special_token("<img>", "image_placeholder")

        #  Some tokens which are used in the preprocessor. We need to keep them
        # here so that the preprocessor works with `tf.data`.
        self._add_special_token("<start_of_image>", "start_of_image_token")
        self._add_special_token("<end_of_image>", "end_of_image_token")

        self._space_padded_tokens = (
            self.start_of_image_token,
            self.end_of_image_token,
            self.image_placeholder,
        )

    @preprocessing_function
    def _tokenize_tf(self, inputs):
        self._maybe_initialized_tf()
        inputs = tf.convert_to_tensor(inputs)
        unbatched = inputs.shape.rank == 0
        if unbatched:
            inputs = tf.expand_dims(inputs, 0)

        inputs = tf.strings.regex_replace(
            inputs, self.start_of_image_token, f" {self.start_of_image_token} "
        )
        inputs = tf.strings.regex_replace(
            inputs, self.end_of_image_token, f" {self.end_of_image_token} "
        )
        inputs = tf.strings.regex_replace(
            inputs, self.image_placeholder, f" {self.image_placeholder} "
        )
        inputs = tf.strings.regex_replace(inputs, "  ", " ")

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
                value=self.end_token_id, dims=tokens.shape.as_list()[0:1] + [1]
            )
            tokens = tf.concat((tokens, eos_tensor), axis=-1)

        # Convert to a dense output if input was a scalar.
        if unbatched:
            tokens = tf.squeeze(tokens, 0)

        return self._cast_tokens_tf(tokens)
