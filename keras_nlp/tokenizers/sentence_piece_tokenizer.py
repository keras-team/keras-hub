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

import base64

import tensorflow as tf
import tensorflow_text as tf_text

from keras_nlp.tokenizers import tokenizer


class SentencePieceTokenizer(tokenizer.Tokenizer):
    """A SentencePiece tokenizer layer.

    This layer provides an implementation of SentencePiece tokenization
    as described in the [SentencePiece paper](https://arxiv.org/abs/1808.06226)
    and the [SentencePiece package](https://pypi.org/project/sentencepiece/).
    The tokenization will run entirely within the Tensorflow graph, and can
    be saved inside a `keras.Model`.

    By default, the layer will output a `tf.RaggedTensor` where the last
    dimension of the output is ragged after whitespace splitting and sub-word
    tokenizing. If `sequence_length` is set, the layer will output a dense
    `tf.Tensor` where all inputs have been padded or truncated to
    `sequence_length`. The output dtype can be controlled via the `dtype`
    argument, which should be either an integer or string type.

    Args:
        model_file: A path to a SentencePiece serialized model file. One
            of `model_file` and `model_bytes` must be set.
        model_bytes: A SentencePiece serialized model byte array. One of
            `model_file` and `model_bytes` must be set.
        sequence_length: If set, the output will be converted to a dense
            tensor and padded/trimmed so all outputs are of sequence_length.

    References:
        - [Kudo and Richardson, 2018](https://arxiv.org/abs/1808.06226)

    Examples:
    ```python
    # Train a SentencePiece vocabulary.
    model = io.BytesIO()
    ds = tf.data.Dataset.from_tensor_slices(["the quick brown fox."])
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=ds.as_numpy_iterator(),
        model_writer=model,
        vocab_size=20,
    )

    # Tokenize inputs
    tokenizer = SentencePieceTokenizer(model_bytes=model.getvalue())
    ds = ds.map(tokenizer)
    ```
    """

    def __init__(
        self,
        model_file: str = None,
        model_bytes: bytes = None,
        sequence_length: int = None,
        **kwargs,
    ) -> None:
        # Check dtype and provide a default.
        if "dtype" not in kwargs or kwargs["dtype"] is None:
            kwargs["dtype"] = tf.int32
        else:
            dtype = tf.dtypes.as_dtype(kwargs["dtype"])
            if not dtype.is_integer and dtype != tf.string:
                raise ValueError(
                    "Output dtype must be an integer type or a string. "
                    f"Received: dtype={dtype}"
                )

        super().__init__(**kwargs)

        if model_file is None and model_bytes is None:
            raise ValueError(
                "One of `model_file` or `model_bytes` must be set. "
                "Received: `model_file=None`, `model_bytes=None`."
            )

        if model_file is not None:
            model_bytes = tf.io.gfile.GFile(model_file, "rb").read()
        if isinstance(model_bytes, str):
            model_bytes = base64.b64decode(model_bytes)

        # Keras cannot serialize a bytestring, so we base64 encode the model
        # byte array for saving.
        self.model_bytes = base64.b64encode(model_bytes).decode("ascii")
        self.sequence_length = sequence_length

        self._sentence_piece = tf_text.SentencepieceTokenizer(
            model=model_bytes,
            out_type=self.compute_dtype,
        )

    def vocabulary_size(self) -> int:
        """Get the size of the tokenizer vocabulary."""
        return self._sentence_piece.vocab_size()

    def id_to_token(self, id: int) -> str:
        """Convert an integer id to a string token."""
        return self._sentence_piece.id_to_string(id)

    def token_to_id(self, token: str) -> int:
        """Convert a string token to an integer id."""
        return self._sentence_piece.string_to_id(token)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                # Ideally the model would be saved as a file asset in
                # the saved model. We have no good way to support this
                # currently, so we save the model string in the config.
                "model_bytes": self.model_bytes,
                "sequence_length": self.sequence_length,
            }
        )
        return config

    def tokenize(self, inputs):
        # Check if Input is Scalar or Not
        if not isinstance(inputs, (tf.Tensor, tf.RaggedTensor)):
            inputs = tf.convert_to_tensor(inputs)
        scalar_input = tf.convert_to_tensor(inputs).shape.rank == 0
        if scalar_input:
            inputs = tf.expand_dims(inputs, 0)

        # Apply word piece and coerce shape for outputs.
        tokens = self._sentence_piece.tokenize(inputs)

        # Convert to a dense output if `sequence_length` is set.
        if self.sequence_length:
            output_shape = tokens.shape.as_list()
            output_shape[-1] = self.sequence_length
            tokens = tokens.to_tensor(shape=output_shape)
        # Convert to a dense output if input in scalar
        if scalar_input:
            tokens = tf.squeeze(tokens, 0)
            tf.ensure_shape(tokens, shape=[self.sequence_length])

        return tokens

    def detokenize(self, inputs):
        return self._sentence_piece.detokenize(inputs)
