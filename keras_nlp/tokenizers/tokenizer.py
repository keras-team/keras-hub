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

from tensorflow import keras


class Tokenizer(keras.layers.Layer):
    """A base class for tokenizer layers.

    The class is intended as a base class when implementing a tokenizer as a
    `keras.layers.Layer`. It contains two new methods `tokenize()` and
    `detokenize()`.

    Subclassers should always implement the `tokenize` method, which will also
    also be the default when invoking the layer directly on inputs

    If a layer does not support detokenization (the tokenization step is not
    reversible), the `detokenize()` method can be skipped.

    Examples:

    ```python
    class WhitespaceTokenizer(keras_nlp.tokenizers.Tokenizer):
        def tokenize(self, inputs):
            return tf.strings.split(inputs).to_tensor()

        def detokenize(self, inputs):
            return tf.strings.reduce_join([inputs], separator=" ", axis=-1)

    tokenizer = WhitespaceTokenizer()
    ```
    """

    def __new__(cls, *args, **kwargs):
        # Wrap the `tokenize` and `detokenize` methods so they route through
        # __call__. This is needed for functional model support.
        obj = super().__new__(cls, *args, **kwargs)
        obj._tokenize_without_call = obj.tokenize
        obj._detokenize_without_call = obj.detokenize
        obj.tokenize = obj._tokenize_with_call
        obj.detokenize = obj._detokenize_with_call
        return obj

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

    def _tokenize_with_call(self, *args, **kwargs):
        return self(*args, mode="tokenize", **kwargs)

    def _detokenize_with_call(self, *args, **kwargs):
        return self(*args, mode="detokenize", **kwargs)

    def call(self, *args, mode="tokenize", training=None, **kwargs):
        if mode == "tokenize":
            return self._tokenize_without_call(*args, **kwargs)
        elif mode == "detokenize":
            return self._detokenize_without_call(*args, **kwargs)
        else:
            raise ValueError(
                f"Unsupported tokenizer mode. Received: mode={mode}"
            )
