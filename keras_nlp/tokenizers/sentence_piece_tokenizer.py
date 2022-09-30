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
import binascii
from typing import List

import tensorflow as tf
from tensorflow import keras

from keras_nlp.tokenizers import tokenizer
from keras_nlp.utils.tf_utils import assert_tf_text_installed
from keras_nlp.utils.tf_utils import tensor_to_string_list

try:
    import tensorflow_text as tf_text
except ImportError:
    tf_text = None


@keras.utils.register_keras_serializable(package="keras_nlp")
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
        proto: Either a `string` path to a SentencePiece proto file, or a
            `bytes` object with a serialized SentencePiece proto. See the
            [SentencePiece repository](https://github.com/google/sentencepiece)
            for more details on the format.
        sequence_length: If set, the output will be converted to a dense
            tensor and padded/trimmed so all outputs are of `sequence_length`.

    References:
        - [Kudo and Richardson, 2018](https://arxiv.org/abs/1808.06226)

    Examples:

    From bytes.
    ```python
    def train_sentence_piece_bytes(ds, size):
        bytes_io = io.BytesIO()
        sentencepiece.SentencePieceTrainer.train(
            sentence_iterator=ds.as_numpy_iterator(),
            model_writer=bytes_io,
            vocab_size=size,
        )
        return bytes_io.getvalue()

    # Train a sentencepiece proto.
    ds = tf.data.Dataset.from_tensor_slices(["the quick brown fox."])
    proto = train_sentence_piece_bytes(ds, 20)
    # Tokenize inputs.
    tokenizer = keras_nlp.tokenizers.SentencePieceTokenizer(proto=proto)
    ds = ds.map(tokenizer)
    ```

    From a file.
    ```python
    def train_sentence_piece_file(ds, path, size):
        with open(path, "wb") as model_file:
            sentencepiece.SentencePieceTrainer.train(
                sentence_iterator=ds.as_numpy_iterator(),
                model_writer=model_file,
                vocab_size=size,
            )

    # Train a sentencepiece proto.
    ds = tf.data.Dataset.from_tensor_slices(["the quick brown fox."])
    proto = train_sentence_piece_file(ds, "model.spm", 20)
    # Tokenize inputs.
    tokenizer = keras_nlp.tokenizers.SentencePieceTokenizer(proto="model.spm")
    ds = ds.map(tokenizer)
    ```
    """

    def __init__(
        self,
        proto,
        sequence_length: int = None,
        **kwargs,
    ) -> None:
        assert_tf_text_installed(self.__class__.__name__)

        # Check dtype and provide a default.
        if "dtype" not in kwargs or kwargs["dtype"] is None:
            kwargs["dtype"] = tf.int32
        else:
            dtype = tf.dtypes.as_dtype(kwargs["dtype"])
            if not dtype.is_integer and dtype != tf.string:
                raise ValueError(
                    "Output dtype must be one of `'string'`, `'int32'`, and "
                    f"`'int64'`. Received: dtype={dtype}"
                )

        super().__init__(**kwargs)

        if isinstance(proto, str):
            # A string could be either a filepath, or a base64 encoded byte
            # array (which we need for serialization). We will heuristically
            # try to distinguish, by checking if a string is both longer and
            # than 2048 characters and valid base64 characters.
            is_base64 = False
            if len(proto) > 2048:
                try:
                    proto_bytes = base64.b64decode(proto, validate=True)
                    is_base64 = True
                except binascii.Error:
                    pass
            if not is_base64:
                proto_bytes = tf.io.gfile.GFile(proto, "rb").read()
        elif isinstance(proto, bytes):
            proto_bytes = proto
        else:
            raise ValueError(
                "SentencePiece `proto` argument should be either a `string` "
                f"filepath or a `bytes` sequence. "
                f"Received unknown type: {type(proto)}"
            )

        self._sentence_piece = tf_text.SentencepieceTokenizer(
            model=proto_bytes,
            out_type=self.compute_dtype,
        )

        # Keras cannot serialize a bytestring, so we base64 encode the model
        # byte array as a string for saving.
        self.proto = base64.b64encode(proto_bytes).decode("ascii")
        self.sequence_length = sequence_length

    def vocabulary_size(self) -> int:
        """Get the size of the tokenizer vocabulary."""
        return int(self._sentence_piece.vocab_size().numpy())

    def get_vocabulary(self) -> List[str]:
        """Get the size of the tokenizer vocabulary."""
        return tensor_to_string_list(
            self._sentence_piece.id_to_string(tf.range(self.vocabulary_size()))
        )

    def id_to_token(self, id: int) -> str:
        """Convert an integer id to a string token."""
        return tensor_to_string_list(self._sentence_piece.id_to_string(id))

    def token_to_id(self, token: str) -> int:
        """Convert a string token to an integer id."""
        return int(self._sentence_piece.string_to_id(token).numpy())

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                # Ideally the model would be saved as a file asset in
                # the saved model. We have no good way to support this
                # currently, so we save the model string in the config.
                "proto": self.proto,
                "sequence_length": self.sequence_length,
            }
        )
        return config

    def tokenize(self, inputs):
        if not isinstance(inputs, (tf.Tensor, tf.RaggedTensor)):
            inputs = tf.convert_to_tensor(inputs)
        scalar_input = inputs.shape.rank == 0
        if scalar_input:
            inputs = tf.expand_dims(inputs, 0)

        tokens = self._sentence_piece.tokenize(inputs)

        # Convert to a dense output if `sequence_length` is set.
        if self.sequence_length:
            output_shape = tokens.shape.as_list()
            output_shape[-1] = self.sequence_length
            tokens = tokens.to_tensor(shape=output_shape)

        # Convert to a dense output if input was a scalar.
        if scalar_input:
            tokens = tf.squeeze(tokens, 0)
            tf.ensure_shape(tokens, shape=[self.sequence_length])

        return tokens

    def detokenize(self, inputs):
        return self._sentence_piece.detokenize(inputs)
