import base64
import binascii
import os

import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.tokenizers import tokenizer
from keras_hub.src.utils.tensor_utils import convert_to_ragged_batch
from keras_hub.src.utils.tensor_utils import is_int_dtype
from keras_hub.src.utils.tensor_utils import is_string_dtype
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import tensor_to_list

try:
    import tensorflow as tf
    import tensorflow_text as tf_text
except ImportError:
    tf = None
    tf_text = None

VOCAB_FILENAME = "vocabulary.spm"


@keras_hub_export("keras_hub.tokenizers.SentencePieceTokenizer")
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
        add_bos: Add beginning of sentence token to the result.
        add_eos: Add end of sentence token to the result. Token is always
            truncated if output is longer than specified `sequence_length`.

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
    tokenizer = keras_hub.tokenizers.SentencePieceTokenizer(proto=proto)
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
    tokenizer = keras_hub.tokenizers.SentencePieceTokenizer(proto="model.spm")
    ds = ds.map(tokenizer)
    ```
    """

    def __init__(
        self,
        proto=None,
        sequence_length=None,
        dtype="int32",
        add_bos=False,
        add_eos=False,
        **kwargs,
    ) -> None:
        if not is_int_dtype(dtype) and not is_string_dtype(dtype):
            raise ValueError(
                "Output dtype must be an integer type or a string. "
                f"Received: dtype={dtype}"
            )

        super().__init__(dtype=dtype, **kwargs)

        self.proto = None
        self.sequence_length = sequence_length
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.set_proto(proto)
        self.file_assets = [VOCAB_FILENAME]

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
            add_bos=self.add_bos,
            add_eos=self.add_eos,
        )
        # Keras cannot serialize a bytestring, so we base64 encode the model
        # byte array as a string for saving.
        self.proto = proto_bytes
        self._update_special_token_ids()

    def vocabulary_size(self):
        """Get the integer size of the tokenizer vocabulary."""
        self._check_vocabulary()
        return int(self._sentence_piece.vocab_size().numpy())

    def get_vocabulary(self):
        """Get the tokenizer vocabulary."""
        self._check_vocabulary()
        return tensor_to_list(
            self._sentence_piece.id_to_string(
                tf.range(int(self._sentence_piece.vocab_size().numpy()))
            )
        )

    def id_to_token(self, id):
        """Convert an integer id to a string token."""
        self._check_vocabulary()
        if id >= self.vocabulary_size() or id < 0:
            raise ValueError(
                f"`id` must be in range [0, {self.vocabulary_size() - 1}]. "
                f"Received: {id}"
            )
        return tensor_to_list(self._sentence_piece.id_to_string(id))

    def token_to_id(self, token):
        """Convert a string token to an integer id."""
        self._check_vocabulary()
        return int(self._sentence_piece.string_to_id(token).numpy())

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "proto": None,  # Save vocabulary via an asset!
                "sequence_length": self.sequence_length,
                "add_bos": self.add_bos,
                "add_eos": self.add_eos,
            }
        )
        return config

    def _check_vocabulary(self):
        if self.proto is None:
            raise ValueError(
                "No vocabulary has been set for SentencePieceTokenizer. Make "
                "sure to pass a `proto` argument when creating the layer."
            )

    @preprocessing_function
    def tokenize(self, inputs):
        self._check_vocabulary()
        inputs = tf.convert_to_tensor(inputs)
        unbatched = inputs.shape.rank == 0
        if unbatched:
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
        if unbatched:
            tokens = tf.squeeze(tokens, 0)
            tf.ensure_shape(tokens, shape=[self.sequence_length])
        return tokens

    @preprocessing_function
    def detokenize(self, inputs):
        self._check_vocabulary()
        inputs, unbatched, rectangular = convert_to_ragged_batch(inputs)
        # tf-text sentencepiece does not handle int64.
        inputs = tf.cast(inputs, "int32")
        outputs = self._sentence_piece.detokenize(inputs)
        if unbatched:
            outputs = tf.squeeze(outputs, 0)
        return outputs

    def compute_output_spec(self, input_spec):
        return keras.KerasTensor(
            input_spec.shape + (self.sequence_length,), dtype=self.compute_dtype
        )
