# Copyright 2023 The KerasNLP Authors
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

from typing import List

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.layers.preprocessing.preprocessing_layer import (
    PreprocessingLayer,
)


@keras_nlp_export("keras_nlp.tokenizers.Tokenizer")
class Tokenizer(PreprocessingLayer):
    """A base class for tokenizer layers.

    Tokenizers in the KerasNLP library should all subclass this layer.
    The class provides two core methods `tokenize()` and `detokenize()` for
    going from plain text to sequences and back. A tokenizer is a subclass of
    `keras.layers.Layer` and can be combined into a `keras.Model`.

    Subclassers should always implement the `tokenize()` method, which will also
    be the default when calling the layer directly on inputs.

    Subclassers can optionally implement the `detokenize()` method if the
    tokenization is reversible. Otherwise, this can be skipped.

    Subclassers should implement `get_vocabulary()`, `vocabulary_size()`,
    `token_to_id()` and `id_to_token()` if applicable. For some simple
    "vocab free" tokenizers, such as a whitespace splitter show below, these
    methods do not apply and can be skipped.

    Example usage:

    ```python
    class WhitespaceSplitterTokenizer(keras_nlp.tokenizers.Tokenizer):
        def tokenize(self, inputs):
            return tf.strings.split(inputs)

        def detokenize(self, inputs):
            return tf.strings.reduce_join(inputs, separator=" ", axis=-1)

    tokenizer = WhitespaceSplitterTokenizer()

    # Tokenize some inputs.
    tokenizer.tokenize("This is a test")

    # Shorthard for `tokenize()`.
    tokenizer("This is a test")

    # Detokenize some outputs.
    tokenizer.detokenize(["This", "is", "a", "test"])
    ```
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tokenize(self, inputs, *args, **kwargs):
        """Transform input tensors of strings into output tokens.

        Args:
            inputs: Input tensor, or dict/list/tuple of input tensors.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        raise NotImplementedError(
            "No implementation of `tokenize()` was found for "
            f"{self.__class__.__name__}. All tokenizers should implement "
            "`tokenize()`."
        )

    def detokenize(self, inputs, *args, **kwargs):
        """Transform tokens back into strings.

        Args:
            inputs: Input tensor, or dict/list/tuple of input tensors.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        raise NotImplementedError(
            "No implementation of `detokenize()` was found for "
            f"{self.__class__.__name__}."
        )

    def get_vocabulary(self) -> List[str]:
        """Get the tokenizer vocabulary as a list of strings terms."""
        raise NotImplementedError(
            "No implementation of `get_vocabulary()` was found for "
            f"{self.__class__.__name__}."
        )

    def vocabulary_size(self) -> int:
        """Returns the total size of the token id space."""
        raise NotImplementedError(
            "No implementation of `vocabulary_size()` was found for "
            f"{self.__class__.__name__}."
        )

    def id_to_token(self, id: int) -> str:
        """Convert an integer id to a string token."""
        raise NotImplementedError(
            "No implementation of `id_to_token()` was found for "
            f"{self.__class__.__name__}."
        )

    def token_to_id(self, token: str) -> int:
        """Convert a string token to an integer id."""
        raise NotImplementedError(
            "No implementation of `token_to_id()` was found for "
            f"{self.__class__.__name__}."
        )

    def call(self, inputs, *args, training=None, **kwargs):
        return self.tokenize(inputs, *args, **kwargs)
