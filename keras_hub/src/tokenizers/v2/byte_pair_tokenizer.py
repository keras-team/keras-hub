import json
import os
import warnings
from typing import Iterable

import keras
import numpy as np
import tokenizers
from keras.src.saving import serialization_lib
from tokenizers import decoders
from tokenizers import models
from tokenizers import pre_tokenizers

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.tokenizers import tokenizer
from keras_hub.src.utils.tensor_utils import is_int_dtype
from keras_hub.src.utils.tensor_utils import is_string_dtype

VOCAB_FILENAME = "vocabulary.json"
MERGES_FILENAME = "merges.txt"

# From Llama3's tokenizer implementation.
SPLIT_PATTERN = (
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| "
    "?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
)


@keras_hub_export("keras_hub.tokenizers.v2.BytePairTokenizer")
class BytePairTokenizer(tokenizer.Tokenizer):
    """Bype-pair encoding tokenizer layer.

    This BPE tokenizer provides the same functionality as the official GPT-2
    tokenizer. Given the same `vocabulary` which maps tokens to ids, and
    `merges` which describes BPE merge rules, it should provide the same output
    as OpenAI implementation (https://github.com/openai/gpt-2/blob/master/src/encoder.py).
    Different from OpenAI, this implementation is graph-compatible, so you can
    use it within a `tf.data` pipeline.

    If input is a batch of strings (rank > 0):
    By default, the layer will output a list of lists. If `sequence_length` is
    set, the layer will output a list of lists where all inputs have been padded
    or truncated to `sequence_length`.
    If input is a scalar string (rank == 0):
    By default, the layer will output a list with static shape. If
    `sequence_length` is set, the output will be a list of shape
    `[sequence_length]`.

    Args:
        vocabulary: string or dict, maps token to integer ids. If it is a
            string, it should be the file path to a json file.
        merges: string or list, contains the merge rule. If it is a string,
            it should be the file path to merge rules. The merge rule file
            should have one merge rule per line.
        sequence_length: int. If set, the output will be
            padded or truncated to the `sequence_length`. Defaults to `None`.
        add_prefix_space: bool. Whether to add an
            initial space to the input. This tokenizer is whitespace aware,
            and will tokenize a word with a leading space differently. Adding
            a prefix space to the first word will cause it to be tokenized
            equivalently to all subsequent words in the sequence.
            Defaults to `False`.
        unsplittable_tokens: list. A list of strings that will
            never be split during the word-level splitting applied before the
            byte-pair encoding. This can be used to ensure special tokens map to
            unique indices in the vocabulary, even if these special tokens
            contain splittable characters such as punctuation. Special tokens
            must still be included in `vocabulary`. Defaults to `None`.

    Examples:

    Tokenize
    >>> vocab = {"butter": 1, "fly": 2}
    >>> merge = ["b u", "t t", "e r", "bu tt", "butt er", "f l", "fl y"]
    >>> tokenizer = keras_hub.tokenizers.BytePairTokenizer(vocab, merge)
    >>> outputs = tokenizer("butterfly")
    >>> np.array(outputs)
    array([1, 2], dtype=int32)
    >>> seq1, seq2 = tokenizer(["butterfly", "butter"])
    >>> np.array(seq1)
    array([1, 2])
    >>> np.array(seq2)
    array([1])
    >>> tokenizer = keras_hub.tokenizers.BytePairTokenizer(
    ...     vocab, merge, sequence_length=2)
    >>> seq1, seq2 = tokenizer(["butterfly", "butter"])
    >>> np.array(seq1)
    array([1, 2], dtype=int32)
    >>> np.array(seq2)
    array([1, 0], dtype=int32)

    Detokenize
    >>> vocab = {"butter": 1, "fly": 2}
    >>> merge = ["b u", "t t", "e r", "bu tt", "butt er", "f l", "fl y"]
    >>> tokenizer = keras_hub.tokenizers.BytePairTokenizer(vocab, merge)
    >>> tokenizer.detokenize([[1, 2]])
    ['butterfly']
    """

    def __init__(
        self,
        vocabulary=None,
        merges=None,
        sequence_length=None,
        add_prefix_space=False,
        unsplittable_tokens=None,
        dtype="int32",
        **kwargs,
    ) -> None:
        if not is_int_dtype(dtype) and not is_string_dtype(dtype):
            raise ValueError(
                "Output dtype must be an integer type or a string. "
                f"Received: dtype={dtype}"
            )

        super().__init__(dtype=dtype, **kwargs)
        self.sequence_length = sequence_length
        self.add_prefix_space = add_prefix_space
        if unsplittable_tokens is None:
            unsplittable_tokens = self.special_tokens
        self.unsplittable_tokens = unsplittable_tokens
        self.file_assets = [VOCAB_FILENAME, MERGES_FILENAME]

        self.set_vocabulary_and_merges(vocabulary, merges)

    def save_assets(self, dir_path):
        vocab_path = os.path.join(dir_path, VOCAB_FILENAME)
        merges_path = os.path.join(dir_path, MERGES_FILENAME)
        with open(vocab_path, "w", encoding="utf-8") as file:
            file.write(json.dumps(dict(self.vocabulary)))
        with open(merges_path, "w", encoding="utf-8") as file:
            for merge in self.merges:
                file.write(f"{merge}\n")

    def load_assets(self, dir_path):
        vocab_path = os.path.join(dir_path, VOCAB_FILENAME)
        merges_path = os.path.join(dir_path, MERGES_FILENAME)
        self.set_vocabulary_and_merges(vocab_path, merges_path)

    def set_vocabulary_and_merges(self, vocabulary, merges):
        """Set the vocabulary and merge rules from data or files."""
        if vocabulary is None or merges is None:
            # Clear vocab related state.
            self.vocabulary = None
            self.merges = None
            return

        if isinstance(vocabulary, str):
            if serialization_lib.in_safe_mode():
                raise ValueError(
                    "Requested the loading of a vocabulary file outside of the "
                    "model archive. This carries a potential risk of loading "
                    "arbitrary and sensitive files and thus it is disallowed "
                    "by default. If you trust the source of the artifact, you "
                    "can override this error by passing `safe_mode=False` to "
                    "the loading function, or calling "
                    "`keras.config.enable_unsafe_deserialization()`. "
                    f"Vocabulary file: '{vocabulary}'"
                )
            with open(vocabulary, "r", encoding="utf-8") as f:
                self.vocabulary = json.load(f)
        elif isinstance(vocabulary, dict):
            self.vocabulary = vocabulary.copy()
        else:
            raise ValueError(
                "Vocabulary must be an file path or dictionary mapping string "
                "token to int ids. Received: "
                f"`type(vocabulary)={type(vocabulary)}`."
            )
        if isinstance(merges, str):
            if serialization_lib.in_safe_mode():
                raise ValueError(
                    "Requested the loading of a merges file outside of the "
                    "model archive. This carries a potential risk of loading "
                    "arbitrary and sensitive files and thus it is disallowed "
                    "by default. If you trust the source of the artifact, you "
                    "can override this error by passing `safe_mode=False` to "
                    "the loading function, or calling "
                    "`keras.config.enable_unsafe_deserialization()`. "
                    f"Merges file: '{merges}'"
                )
            with open(merges, encoding="utf-8") as f:
                merges = [bp.rstrip() for bp in f]
        elif isinstance(merges, Iterable):
            merges = list(merges)
        else:
            raise ValueError(
                "Merges must be a file path or a list of merge rules. "
                f"Received: `type(merges)={type(merges)}`"
            )
        self.merges = merges
        _merges = []
        for merge in merges:
            a, b = merge.split(" ")
            if a not in self.vocabulary or b not in self.vocabulary:
                warnings.warn(
                    f"Merge pair ({a}, {b}) contains a token not in the "
                    "vocabulary. Skipping."
                )
                continue
            _merges.append((a, b))

        self._tokenizer = tokenizers.Tokenizer(
            models.BPE(vocab=self.vocabulary, merges=_merges)
        )
        if self.unsplittable_tokens:
            self._tokenizer.add_special_tokens(self.unsplittable_tokens)
        # Ensure the implementation matches Llama3's tokenizer behavior.
        self._tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(
                    pattern=SPLIT_PATTERN, behavior="isolated"
                ),
                pre_tokenizers.ByteLevel(
                    add_prefix_space=self.add_prefix_space, use_regex=False
                ),
            ]
        )
        self._tokenizer.decoder = decoders.ByteLevel()
        self._update_special_token_ids()

    def get_vocabulary(self):
        """Get the tokenizer vocabulary as a list of strings tokens."""
        self._check_vocabulary()
        return self._tokenizer.get_vocab().keys()

    def vocabulary_size(self):
        """Get the integer size of the tokenizer vocabulary."""
        self._check_vocabulary()
        return self._tokenizer.get_vocab_size()

    def id_to_token(self, id):
        """Convert an integer id to a string token."""
        self._check_vocabulary()
        try:
            token = self._tokenizer.id_to_token(id)
        except OverflowError:
            token = None
        if token is None:
            raise ValueError(f"Id {id} is out of vocabulary range.")
        return token

    def token_to_id(self, token):
        """Convert a string token to an integer id."""
        self._check_vocabulary()
        token_id = self._tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"Token '{token}' is not in the vocabulary.")
        return token_id

    def _check_vocabulary(self):
        if self.vocabulary is None:
            raise ValueError(
                "No vocabulary has been set for BytePairTokenizer. Make sure "
                "to pass `vocabulary` and `merges` arguments when creating the "
                "layer."
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
        outputs = self._tokenizer.encode_batch(inputs)
        if is_int_dtype(self.compute_dtype):
            batched_tokens = [o.ids for o in outputs]
        else:
            batched_tokens = [o.tokens for o in outputs]

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

    def detokenize(self, inputs):
        self._check_vocabulary()
        inputs, batched = self._canonicalize_detokenize_inputs(inputs)
        outputs = self._tokenizer.decode_batch(inputs)
        if not batched:
            outputs = outputs[0]
        return outputs

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
                "sequence_length": self.sequence_length,
                "add_prefix_space": self.add_prefix_space,
                "unsplittable_tokens": self.unsplittable_tokens,
            }
        )
        return config
