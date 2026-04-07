import re

import tensorflow as tf

from keras_hub.src.tokenizers.tokenizer import Tokenizer
from keras_hub.src.utils.tensor_utils import convert_to_ragged_batch
from keras_hub.src.utils.tensor_utils import is_int_dtype
from keras_hub.src.utils.tensor_utils import is_string_dtype
from keras_hub.src.utils.tensor_utils import preprocessing_function


class MockGemma4Tokenizer(Tokenizer):
    def __init__(
        self,
        proto=None,
        sequence_length=None,
        dtype="int32",
        add_bos=False,
        add_eos=False,
        **kwargs,
    ):
        if not is_int_dtype(dtype) and not is_string_dtype(dtype):
            raise ValueError(
                "Output dtype must be an integer type or a string. "
                f"Received: dtype={dtype}"
            )

        super().__init__(dtype=dtype, **kwargs)

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
        ]
        self.string_to_id = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                self.vocabulary, list(range(len(self.vocabulary)))
            ),
            default_value=3,
        )
        self.id_to_string = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                list(range(len(self.vocabulary))), self.vocabulary
            ),
            default_value="<unk>",
        )

        # Standard tokens.
        self._add_special_token("<bos>", "start_token")
        self._add_special_token("<eos>", "end_token")
        self._add_special_token("<pad>", "pad_token")

        # Image placeholder token.
        self._add_special_token("<|image|>", "image_placeholder")

        # Boundary tokens used in the preprocessor.
        self._add_special_token("<|image>", "start_of_image_token")
        self._add_special_token("<image|>", "end_of_image_token")

        self.sequence_length = sequence_length
        self.add_bos = add_bos
        self.add_eos = add_eos

    def vocabulary_size(self):
        return len(self.vocabulary)

    def get_vocabulary(self):
        return self.vocabulary

    def id_to_token(self, id):
        return self.vocabulary[id]

    def token_to_id(self, token):
        return self.vocabulary.index(token)

    @preprocessing_function
    def tokenize(self, inputs):
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

        return tokens

    @preprocessing_function
    def detokenize(self, inputs):
        inputs, unbatched, rectangular = convert_to_ragged_batch(inputs)
        inputs = tf.cast(inputs, "int32")

        outputs = self.id_to_string.lookup(inputs)
        outputs = tf.strings.reduce_join(outputs, axis=-1, separator=" ")

        for token in [
            self.start_token,
            self.end_token,
            self.pad_token,
        ]:
            outputs = tf.strings.regex_replace(outputs, token, "")

        outputs = tf.strings.strip(outputs)

        if unbatched:
            outputs = tf.squeeze(outputs, 0)
        return outputs

    def __call__(self, inputs):
        return self.tokenize(inputs)
