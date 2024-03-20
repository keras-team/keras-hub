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

import base64
import binascii
import os
from typing import List

import tensorflow as tf

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.tokenizers import tokenizer
from keras_nlp.utils.preset_utils import check_preset_class
from keras_nlp.utils.preset_utils import load_from_preset
from keras_nlp.utils.python_utils import classproperty
from keras_nlp.utils.python_utils import format_docstring
from keras_nlp.utils.tensor_utils import assert_tf_text_installed
from keras_nlp.utils.tensor_utils import convert_to_ragged_batch
from keras_nlp.utils.tensor_utils import is_int_dtype
from keras_nlp.utils.tensor_utils import is_string_dtype
from keras_nlp.utils.tensor_utils import tensor_to_list

try:
    import tensorflow_text as tf_text
except ImportError:
    tf_text = None


VOCAB_FILENAME = "vocabulary.spm"


@keras_nlp_export("keras_nlp.tokenizers.SentencePieceTokenizer")
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
        proto=None,
        sequence_length: int = None,
        dtype="int32",
        **kwargs,
    ) -> None:
        assert_tf_text_installed(self.__class__.__name__)

        if not is_int_dtype(dtype) and not is_string_dtype(dtype):
            raise ValueError(
                "Output dtype must be an integer type or a string. "
                f"Received: dtype={dtype}"
            )

        super().__init__(dtype=dtype, **kwargs)

        self.proto = None
        self.sequence_length = sequence_length
        self.set_proto(proto)

    def save_assets(self, dir_path):
        path = os.path.join(dir_path, VOCAB_FILENAME)
        with open(path, "wb") as file:
            file.write(self.proto)

    def load_assets(self, dir_path):
        path = os.path.join(dir_path, VOCAB_FILENAME)
        self.set_proto(path)

    def set_proto(self, proto):
        if proto is None:
            self.proto = None
            self._sentence_piece = None
            return

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
                proto_bytes = open(proto, "rb").read()
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
        self.proto = proto_bytes

    def vocabulary_size(self) -> int:
        """Get the size of the tokenizer vocabulary."""
        self._check_vocabulary()
        return int(self._sentence_piece.vocab_size().numpy())

    def get_vocabulary(self) -> List[str]:
        """Get the tokenizer vocabulary."""
        self._check_vocabulary()
        return tensor_to_list(
            self._sentence_piece.id_to_string(
                tf.range(int(self._sentence_piece.vocab_size().numpy()))
            )
        )

    def id_to_token(self, id: int) -> str:
        """Convert an integer id to a string token."""
        self._check_vocabulary()
        if id >= self.vocabulary_size() or id < 0:
            raise ValueError(
                f"`id` must be in range [0, {self.vocabulary_size() - 1}]. "
                f"Received: {id}"
            )
        return tensor_to_list(self._sentence_piece.id_to_string(id))

    def token_to_id(self, token: str) -> int:
        """Convert a string token to an integer id."""
        self._check_vocabulary()
        return int(self._sentence_piece.string_to_id(token).numpy())

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "proto": None,  # Save vocabulary via an asset!
                "sequence_length": self.sequence_length,
            }
        )
        return config

    def _check_vocabulary(self):
        if self.proto is None:
            raise ValueError(
                "No vocabulary has been set for SentencePieceTokenizer. Make "
                "sure to pass a `proto` argument when creating the layer."
            )

    def tokenize(self, inputs):
        self._check_vocabulary()
        if not isinstance(inputs, (tf.Tensor, tf.RaggedTensor)):
            inputs = tf.convert_to_tensor(inputs)
        scalar_input = inputs.shape.rank == 0
        if scalar_input:
            inputs = tf.expand_dims(inputs, 0)

        if self._sentence_piece is None:
            raise ValueError(
                "No vocabulary has been set for SentencePieceTokenizer. Make "
                "sure to pass a `vocabulary` argument when creating the layer."
            )

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
        self._check_vocabulary()
        inputs, unbatched, _ = convert_to_ragged_batch(inputs)
        # tf-text sentencepiece does not handle int64.
        inputs = tf.cast(inputs, "int32")
        outputs = self._sentence_piece.detokenize(inputs)
        if unbatched:
            outputs = tf.squeeze(outputs, 0)
        return outputs

    @classproperty
    def presets(cls):
        return {}

    @classmethod
    def from_preset(
        cls,
        preset,
        **kwargs,
    ):
        """Instantiate {{model_name}} tokenizer from preset vocabulary.

        Args:
            preset: string. Must be one of "{{preset_names}}".

        Examples:
        ```python
        # Load a preset tokenizer.
        tokenizer = {{model_name}}.from_preset("{{example_preset_name}}")

        # Tokenize some input.
        tokenizer("The quick brown fox tripped.")

        # Detokenize some input.
        tokenizer.detokenize([5, 6, 7, 8, 9])
        ```
        """
        # We support short IDs for official presets, e.g. `"bert_base_en"`.
        # Map these to a Kaggle Models handle.
        if preset in cls.presets:
            preset = cls.presets[preset]["kaggle_handle"]

        config_file = "tokenizer.json"
        check_preset_class(preset, cls, config_file=config_file)
        return load_from_preset(
            preset,
            config_file=config_file,
            config_overrides=kwargs,
        )

    def __init_subclass__(cls, **kwargs):
        # Use __init_subclass__ to setup a correct docstring for from_preset.
        super().__init_subclass__(**kwargs)

        # If the subclass does not define from_preset, assign a wrapper so that
        # each class can have a distinct docstring.
        if "from_preset" not in cls.__dict__:

            def from_preset(calling_cls, *args, **kwargs):
                return super(cls, calling_cls).from_preset(*args, **kwargs)

            cls.from_preset = classmethod(from_preset)

        # Format and assign the docstring unless the subclass has overridden it.
        if cls.from_preset.__doc__ is None:
            cls.from_preset.__func__.__doc__ = (
                SentencePieceTokenizer.from_preset.__doc__
            )
            format_docstring(
                model_name=cls.__name__,
                example_preset_name=next(iter(cls.presets), ""),
                preset_names='", "'.join(cls.presets),
            )(cls.from_preset.__func__)
