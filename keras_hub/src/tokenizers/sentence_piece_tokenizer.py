import base64
import binascii
import os

import keras
import numpy as np
from keras.src.saving import serialization_lib

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.tokenizers import tokenizer
from keras_hub.src.utils.tensor_utils import assert_tf_libs_installed
from keras_hub.src.utils.tensor_utils import convert_to_ragged_batch
from keras_hub.src.utils.tensor_utils import in_tf_function
from keras_hub.src.utils.tensor_utils import is_int_dtype
from keras_hub.src.utils.tensor_utils import is_string_dtype
from keras_hub.src.utils.tensor_utils import preprocessing_function

try:
    import tensorflow as tf
except ImportError:
    tf = None
try:
    import tensorflow_text as tf_text
except ImportError:
    tf_text = None
try:
    import sentencepiece as spm
except ImportError:
    spm = None

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

        _allow_python_workflow = kwargs.pop("_allow_python_workflow", True)
        super().__init__(
            dtype=dtype, _allow_python_workflow=_allow_python_workflow, **kwargs
        )

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

    def _set_proto_tf(self, proto):
        assert_tf_libs_installed(self.__class__.__name__)
        self._sentence_piece = tf_text.SentencepieceTokenizer(
            model=proto,
            out_type=self.compute_dtype,
            add_bos=self.add_bos,
            add_eos=self.add_eos,
        )

    def _set_proto_spm(self, proto):
        self._sentence_piece_spm = spm.SentencePieceProcessor()
        self._sentence_piece_spm.Init(
            model_proto=proto,
            out_type=str if is_string_dtype(self.compute_dtype) else int,
            add_bos=self.add_bos,
            add_eos=self.add_eos,
            alpha=1.0,
        )

    def set_proto(self, proto):
        if proto is None:
            self.proto = None
            # _sentence_piece
            self._sentence_piece = None
            # _sentence_piece_spm
            self._sentence_piece_spm = None
            self._vocabulary = None
            self._vocabulary_size = None
            self._token_to_id_map = None
            self._unk_token_id = None
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
                if serialization_lib.in_safe_mode():
                    raise ValueError(
                        "Requested the loading of a proto file outside of "
                        "the model archive. This carries a potential risk of "
                        "loading arbitrary and sensitive files and thus it is "
                        "disallowed by default. If you trust the source of the "
                        "artifact, you can override this error by passing "
                        "`safe_mode=False` to the loading function, or calling "
                        "`keras.config.enable_unsafe_deserialization()`. "
                        f"Proto file: '{proto}'"
                    )
                proto_bytes = open(proto, "rb").read()
        elif isinstance(proto, bytes):
            proto_bytes = proto
        else:
            raise ValueError(
                "SentencePiece `proto` argument should be either a `string` "
                f"filepath or a `bytes` sequence. "
                f"Received unknown type: {type(proto)}"
            )

        # When using `SentencePieceTokenizer` with `tf.data`, it must be built
        # outside the `tf.data` pipeline. So we always call `_set_proto_tf`.
        try:
            self._set_proto_tf(proto_bytes)
        except ImportError:
            pass

        # Use native sentencepiece to extract vocabulary metadata.
        # This avoids TF ops (.numpy()), making this code safe in
        # any execution context.
        if spm is None:
            raise ImportError(
                "SentencePieceTokenizer requires the `sentencepiece` package. "
                "Please install it with `pip install sentencepiece`."
            )
        self._set_proto_spm(proto_bytes)

        # Cache metadata for fast access.
        self._vocabulary_size = int(self._sentence_piece_spm.vocab_size())
        self._vocabulary = list(
            self._sentence_piece_spm.IdToPiece(
                list(range(self._vocabulary_size))
            )
        )
        self._token_to_id_map = {
            token: id for id, token in enumerate(self._vocabulary)
        }
        self._unk_token_id = self._sentence_piece_spm.unk_id()

        # Keras cannot serialize a bytestring, so we base64 encode the model
        # byte array as a string for saving.
        self.proto = proto_bytes
        self._update_special_token_ids()

    def _check_vocabulary(self):
        if self.proto is None:
            raise ValueError(
                "No vocabulary has been set for SentencePieceTokenizer. Make "
                "sure to pass a `proto` argument when creating the layer."
            )

    def _maybe_initialized_tf(self):
        if getattr(self, "_sentence_piece", None) is None:
            self._set_proto_tf(self.proto)

    def _maybe_initialized_spm(self):
        if getattr(self, "_sentence_piece_spm", None) is None:
            self._set_proto_spm(self.proto)

    def vocabulary_size(self):
        """Get the integer size of the tokenizer vocabulary."""
        self._check_vocabulary()
        return self._vocabulary_size

    def get_vocabulary(self):
        """Get the tokenizer vocabulary."""
        self._check_vocabulary()
        return list(self._vocabulary)

    def _id_to_token_tf(self, id):
        self._maybe_initialized_tf()
        return self._vocabulary[id]

    def _id_to_token_spm(self, id):
        self._maybe_initialized_spm()
        return self._sentence_piece_spm.IdToPiece(id)

    def id_to_token(self, id):
        """Convert an integer id to a string token."""
        self._check_vocabulary()
        if id >= self.vocabulary_size() or id < 0:
            raise ValueError(
                f"`id` must be in range [0, {self.vocabulary_size() - 1}]. "
                f"Received: {id}"
            )
        if not self._allow_python_workflow or in_tf_function():
            return self._id_to_token_tf(id)
        else:
            return self._id_to_token_spm(id)

    def token_to_id(self, token):
        """Convert a string token to an integer id."""
        self._check_vocabulary()
        if hasattr(token, "numpy"):
            token = token.numpy()
        if isinstance(token, bytes):
            token = token.decode("utf-8")
        # Return unk_id for unknown tokens, matching the original
        # SentencePiece `string_to_id()` behavior.
        return self._token_to_id_map.get(token, self._unk_token_id)

    @preprocessing_function
    def _tokenize_tf(self, inputs):
        self._maybe_initialized_tf()
        inputs = tf.convert_to_tensor(inputs)
        unbatched = inputs.shape.rank == 0
        if unbatched:
            inputs = tf.expand_dims(inputs, 0)

        tokens = self._sentence_piece.tokenize(inputs)

        # Convert to a dense output if `sequence_length` is set.
        if self.sequence_length:
            output_shape = tokens.shape.as_list()
            output_shape[-1] = self.sequence_length
            pad_token_id = getattr(self, "pad_token_id", 0)
            tokens = tokens.to_tensor(
                shape=output_shape, default_value=pad_token_id
            )

        # Convert to a dense output if input was a scalar.
        if unbatched:
            tokens = tf.squeeze(tokens, 0)
            tf.ensure_shape(tokens, shape=[self.sequence_length])
        return tokens

    def _tokenize_spm(self, inputs):
        self._maybe_initialized_spm()

        def _canonicalize_tokenize_inputs(inputs):
            if isinstance(inputs, (str, bytes)):
                if isinstance(inputs, bytes):
                    inputs = inputs.decode("utf-8")
                return [inputs], False
            elif isinstance(inputs, (tuple, list)):
                inputs = list(inputs)
                for i in range(len(inputs)):
                    if isinstance(inputs[i], bytes):
                        inputs[i] = inputs[i].decode("utf-8")
                    elif not isinstance(inputs[i], str):
                        raise ValueError(
                            "If a list or tuple is provided as input, all "
                            f"elements must be strings. Received: {inputs}"
                        )
                return inputs, True
            elif (
                isinstance(inputs, np.ndarray)
                or keras.ops.is_tensor(inputs)
                or (tf is not None and isinstance(inputs, tf.Tensor))
            ):
                inputs = keras.ops.convert_to_numpy(inputs)
                if inputs.ndim == 0:
                    val = inputs.item()
                    if isinstance(val, bytes):
                        val = val.decode("utf-8")
                    return [val], False
                elif inputs.ndim == 1:
                    val_list = inputs.tolist()
                    for i in range(len(val_list)):
                        if isinstance(val_list[i], bytes):
                            val_list[i] = val_list[i].decode("utf-8")
                        elif not isinstance(val_list[i], str):
                            raise ValueError(
                                "If a array is provided as input, all elements "
                                f"must be strings. Received: {inputs}"
                            )
                    return val_list, True
                else:
                    raise ValueError(
                        f"Array must be 0 or 1 dimensional, got {inputs.shape}."
                    )
            else:
                raise ValueError(
                    "Input should be a string or a list of strings. "
                    f"Received: {inputs}"
                )

        inputs, batched = _canonicalize_tokenize_inputs(inputs)
        batched_tokens = self._sentence_piece_spm.Encode(
            inputs, add_bos=self.add_bos, add_eos=self.add_eos
        )

        # Convert to a dense output if `sequence_length` is set.
        if self.sequence_length:
            # Truncate sequences to `sequence_length`.
            batched_tokens = [
                tokens[: self.sequence_length] for tokens in batched_tokens
            ]
            # Pad sequences to `sequence_length`.
            pad_token_id = getattr(self, "pad_token_id", 0)
            batched_tokens = [
                tokens + [pad_token_id] * (self.sequence_length - len(tokens))
                for tokens in batched_tokens
            ]

        if not batched:
            batched_tokens = batched_tokens[0]
        return batched_tokens

    def tokenize(self, inputs):
        self._check_vocabulary()
        if not self._allow_python_workflow or in_tf_function():
            return self._tokenize_tf(inputs)
        else:
            return self._tokenize_spm(inputs)

    @preprocessing_function
    def _detokenize_tf(self, inputs):
        self._maybe_initialized_tf()
        inputs, unbatched, rectangular = convert_to_ragged_batch(inputs)
        # tf-text sentencepiece does not handle int64.
        inputs = tf.cast(inputs, "int32")
        outputs = self._sentence_piece.detokenize(inputs)
        if unbatched:
            outputs = tf.squeeze(outputs, 0)
        return outputs

    def _canonicalize_detokenize_spm_inputs(self, inputs):
        if tf is not None and isinstance(inputs, (tf.Tensor, tf.RaggedTensor)):
            if isinstance(inputs, tf.RaggedTensor):
                inputs = inputs.to_list()
            else:
                inputs = np.array(inputs)
        is_batched = True
        if isinstance(inputs, int):
            inputs = [[inputs]]
            is_batched = False
        elif isinstance(inputs, (tuple, list)):
            if not inputs or isinstance(inputs[0], int):
                # Unbatched list of ints.
                inputs = [list(inputs)]
                is_batched = False
            else:
                # Batched list of lists of ints.
                inputs = [list(seq) for seq in inputs]
        elif isinstance(inputs, np.ndarray) or keras.ops.is_tensor(inputs):
            inputs = keras.ops.convert_to_numpy(inputs)
            if inputs.ndim == 0:
                inputs = [[inputs.item()]]
                is_batched = False
            elif inputs.ndim == 1:
                inputs = [inputs.tolist()]
                is_batched = False
            elif inputs.ndim == 2:
                inputs = inputs.tolist()
            else:
                raise ValueError(
                    f"Array must be 0, 1 or 2 dimensional, got {inputs.shape}."
                )
        else:
            raise ValueError(
                "Input should be an integer, a list of integers, backend "
                f"tensor or numpy array. Received: {inputs}"
            )
        return inputs, is_batched

    def _chunk_by_special_tokens(self, seq, special_ids=None):
        if special_ids is None:
            try:
                special_ids = set(self.special_token_ids)
            except ValueError:
                special_ids = set()

        current_chunk = []
        for token_id in seq:
            if token_id in special_ids:
                if current_chunk:
                    yield False, current_chunk
                    current_chunk = []
                yield True, [token_id]
            else:
                current_chunk.append(token_id)
        if current_chunk:
            yield False, current_chunk

    def _decode_with_special_tokens(self, inputs):
        if not hasattr(self, "_special_token_attrs") or not self._special_token_attrs:
            return self._sentence_piece_spm.Decode(inputs)

        try:
            special_ids = set(self.special_token_ids)
        except ValueError:
            special_ids = set()

        if not special_ids:
            return self._sentence_piece_spm.Decode(inputs)

        outputs = []
        for seq in inputs:
            words = []
            for is_special, chunk in self._chunk_by_special_tokens(seq, special_ids):
                if is_special:
                    words.append(self.id_to_token(chunk[0]))
                else:
                    words.append(self._sentence_piece_spm.Decode(chunk))
            outputs.append("".join(words))
        return outputs

    def _detokenize_spm(self, inputs):
        self._maybe_initialized_spm()
        inputs, batched = self._canonicalize_detokenize_spm_inputs(inputs)
        outputs = self._decode_with_special_tokens(inputs)
        if not batched:
            outputs = outputs[0]
        return outputs

    def detokenize(self, inputs):
        self._check_vocabulary()
        if not self._allow_python_workflow or in_tf_function():
            return self._detokenize_tf(inputs)
        else:
            return self._detokenize_spm(inputs)

    def call(self, inputs, *args, training=None, **kwargs):
        return self.tokenize(inputs, *args, **kwargs)

    def compute_output_spec(self, input_spec):
        return keras.KerasTensor(
            input_spec.shape + (self.sequence_length,), dtype=self.compute_dtype
        )

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
