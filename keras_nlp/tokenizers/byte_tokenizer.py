# Copyright 2022 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Byte Tokenizer."""

from typing import Any
from typing import Dict
from typing import List

import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text

from keras_nlp.tokenizers import tokenizer


class ByteTokenizer(tokenizer.Tokenizer):
    """Raw byte tokenizer.

    This tokenizer is a vocabulary-free tokenizer which will tokenize text as
    as raw bytes from [0, 256).

    By default, the layer will output a `tf.RaggedTensor` where the last
    dimension of the output is ragged after whitespace splitting and sub-word
    tokenizing. If `sequence_length` is set, the layer will output a dense
    `tf.Tensor` where all inputs have been padded or truncated to
    `sequence_length`. The output dtype can be controlled via the `dtype`
    argument, which should be an integer type (tf.int16, tf.int32, etc.).

    Args:
    lowercase: boolean. If True, the input text will be converted to lowercase
        before tokenization.
    sequence_length: int. If set, the output will be converted to a dense
        tensor and padded/trimmed so all outputs are of sequence_length.
    normalization_form: string. One of the following values: (None, "NFC",
        "NFKC", "NFD", "NFKD"). If set, every UTF-8 string in the input tensor
        text will be normalized to the given form before tokenizing.
    errors: string. One of ("strict", "replace", "ignore"). Defaults to
        "replace".Specifies the `detokenize()` behaviour when an invalid byte
        sequence is encountered (same behaviour as
        https://www.tensorflow.org/api_docs/python/tf/strings/unicode_transcode).
    replacement_char: int. Defaults to 65533. The replacement character to use
        when an invalid byte sequence is encountered and when `errors` is set to
        "replace" (same behaviour as
        https://www.tensorflow.org/api_docs/python/tf/strings/unicode_transcode).
    """

    def __init__(
        self,
        lowercase: bool = True,
        sequence_length: int = None,
        normalization_form: str = None,
        errors: str = "replace",
        replacement_char: int = 65533,
        **kwargs,
    ):
        # Check dtype and provide a default.
        if "dtype" not in kwargs or kwargs["dtype"] is None:
            kwargs["dtype"] = tf.int32
        else:
            dtype = tf.dtypes.as_dtype(kwargs["dtype"])
            if not dtype.is_integer:
                raise ValueError(
                    "Output dtype must be an integer type. "
                    f"Received: dtype={dtype}"
                )

        # Check normalization_form.
        if normalization_form not in (None, "NFC", "NFKC", "NFD", "NFKD"):
            raise ValueError(
                '`normalization_form` must be one of None, "NFC", "NFKC", '
                '"NFD", "NFKD". Received: normalization_form='
                f"{normalization_form}"
            )

        # Check errors.
        if errors not in ("strict", "replace", "ignore"):
            raise ValueError(
                '`errors` must be one of "strict", "replace", "ignore" '
                f"Received: errors={errors}"
            )

        super().__init__(**kwargs)

        self._dtype = kwargs["dtype"]
        self._vocab = [i.tobytes() for i in np.arange(256, dtype=np.uint8)]
        self._lowercase = lowercase
        self._sequence_length = sequence_length
        self._normalization_form = normalization_form
        self._errors = errors
        self._replacement_char = replacement_char

    def get_vocabulary(self) -> List[str]:
        """Get the tokenizer vocabulary as a list of strings tokens."""
        return self._vocab

    def vocabulary_size(self) -> int:
        """Get the size of the tokenizer vocabulary."""
        return len(self._vocab)

    def id_to_token(self, id: int) -> str:
        """Convert an integer id to a string token."""
        return self._vocab[id]

    def token_to_id(self, token: str) -> int:
        """Convert a string token to an integer id."""
        return self._vocab.index(token)

    def tokenize(self, inputs):
        # Optional: Lowercase the input.
        if self._lowercase:
            inputs = tf_text.case_fold_utf8(inputs)

        # Optional: Normalize unicode.
        if self._normalization_form is not None:
            inputs = tf_text.normalize_utf8(inputs, self._normalization_form)

        # Tokenize input strings.
        tokens = tf.strings.bytes_split(inputs)
        tokens = tf.squeeze(
            tf.ragged.map_flat_values(tf.io.decode_raw, tokens, tf.uint8), -1
        )
        tokens = tf.cast(tokens, self._dtype)

        # Convert to a dense output if `sequence_length` is set.
        if self._sequence_length:
            output_shape = tokens.shape.as_list()
            output_shape[-1] = self._sequence_length
            tokens = tokens.to_tensor(shape=output_shape)
        return tokens

    def detokenize(self, inputs):
        idx_to_char_lst = tf.constant(self._vocab)
        decoded = tf.strings.reduce_join(
            tf.gather(idx_to_char_lst, inputs), axis=-1
        )
        decoded = tf.strings.unicode_transcode(
            decoded,
            "UTF-8",
            "UTF-8",
            errors=self._errors,
            replacement_char=self._replacement_char,
        )
        return decoded

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update(
            {
                "lowercase": self._lowercase,
                "sequence_length": self._sequence_length,
                "normalization_form": self._normalization_form,
                "errors": self._errors,
                "replacement_char": self._replacement_char,
            }
        )
        return config
