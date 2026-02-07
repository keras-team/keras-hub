import base64
import binascii
import os

import keras
import numpy as np
import sentencepiece as spm
from keras.src.saving import serialization_lib

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.tokenizers import tokenizer
from keras_hub.src.utils.tensor_utils import is_int_dtype
from keras_hub.src.utils.tensor_utils import is_string_dtype

VOCAB_FILENAME = "vocabulary.spm"


@keras_hub_export("keras_hub.tokenizers.v2.SentencePieceTokenizer")
class SentencePieceTokenizer(tokenizer.Tokenizer):
    """A SentencePiece tokenizer layer.

    This layer provides an implementation of SentencePiece tokenization
    as described in the [SentencePiece paper](https://arxiv.org/abs/1808.06226)
    and the [SentencePiece package](https://pypi.org/project/sentencepiece/).
    The tokenization will be backend agnostic by using the pure Python API of
    the SentencePiece package.

    By default, the layer will output a list of ints where the last
    after whitespace splitting and sub-word tokenizing. If `sequence_length` is
    set, the layer will output a same-length lists where all inputs have been
    padded or truncated to `sequence_length`. The output dtype can be
    controlled via the `dtype` argument, which should be either an integer or
    string type.

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

        self._sentence_piece = spm.SentencePieceProcessor()
        self._sentence_piece.Init(
            model_proto=proto_bytes,
            out_type=str if is_string_dtype(self.compute_dtype) else int,
            add_bos=self.add_bos,
            add_eos=self.add_eos,
            alpha=1.0,
        )

        # Keras cannot serialize a bytestring, so we base64 encode the model
        # byte array as a string for saving.
        self.proto = proto_bytes
        self._update_special_token_ids()

    def vocabulary_size(self):
        """Get the integer size of the tokenizer vocabulary."""
        self._check_vocabulary()
        return int(self._sentence_piece.vocab_size())

    def get_vocabulary(self):
        """Get the tokenizer vocabulary."""
        self._check_vocabulary()
        return list(
            self._sentence_piece.IdToPiece(
                list(range(int(self._sentence_piece.vocab_size())))
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
        return self._sentence_piece.IdToPiece(id)

    def token_to_id(self, token):
        """Convert a string token to an integer id."""
        self._check_vocabulary()
        return int(self._sentence_piece.PieceToId(token))

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

    def _canonicalize_tokenize_inputs(self, inputs):
        if isinstance(inputs, str):
            return [inputs], False
        elif isinstance(inputs, (tuple, list)):
            if not all(isinstance(i, str) for i in inputs):
                raise ValueError(
                    "If a list or tuple is provided as input, all elements "
                    "must be strings. "
                    f"Received: {inputs}"
                )
            return list(inputs), True
        else:
            raise ValueError(
                "Input should be a string or a list of strings. "
                f"Received: {inputs}"
            )

    def _canonicalize_detokenize_inputs(self, inputs):
        if isinstance(inputs, int):
            return [inputs], False
        elif isinstance(inputs, (tuple, list)):
            return list(inputs), True
        elif isinstance(inputs, np.ndarray) or keras.ops.is_tensor(inputs):
            inputs = keras.ops.convert_to_numpy(inputs)
            if inputs.ndim == 0:  # scalar
                inputs = [inputs.item()]
            elif inputs.ndim == 1:
                inputs = inputs.tolist()
            else:
                raise ValueError(
                    f"Array must be 0 or 1 dimensional, got {inputs.shape}."
                )
            return inputs, True
        else:
            raise ValueError(
                "Input should be an integer, a list of integers, backend "
                f"tensor or numpy array. Received: {inputs}"
            )

    def tokenize(self, inputs):
        self._check_vocabulary()
        inputs, batched = self._canonicalize_tokenize_inputs(inputs)

        if self._sentence_piece is None:
            raise ValueError(
                "No vocabulary has been set for SentencePieceTokenizer. Make "
                "sure to pass a `vocabulary` argument when creating the layer."
            )

        batched_tokens = self._sentence_piece.Encode(
            inputs, add_bos=self.add_bos, add_eos=self.add_eos
        )

        # Convert to a dense output if `sequence_length` is set.
        if self.sequence_length:
            # Truncate sequences to `sequence_length`.
            batched_tokens = [
                tokens[: self.sequence_length] for tokens in batched_tokens
            ]
            # Pad sequences to `sequence_length`.
            batched_tokens = [
                tokens
                + [self._sentence_piece.pad_id()]
                * (self.sequence_length - len(tokens))
                for tokens in batched_tokens
            ]

        if not batched:
            batched_tokens = batched_tokens[0]
        return batched_tokens

    def detokenize(self, inputs):
        self._check_vocabulary()
        inputs, batched = self._canonicalize_detokenize_inputs(inputs)
        outputs = self._sentence_piece.Decode(inputs)
        if not batched:
            outputs = outputs[0]
        return outputs

    def call(self, inputs, *args, training=None, **kwargs):
        return self.tokenize(inputs, *args, **kwargs)

    def compute_output_spec(self, input_spec):
        return keras.KerasTensor(
            input_spec.shape + (self.sequence_length,), dtype=self.compute_dtype
        )
