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

"""Byte-pair encoder implementation.

This file implements the same logic as openai BPE:
https://github.com/openai/gpt-2/blob/master/src/encoder.py,
but is TF graph compatible.
"""

import json
import os
from typing import Iterable

import keras
import regex as re

try:
    import tensorflow as tf
except ImportError:
    raise ImportError(
        "To use `keras_nlp`, please install Tensorflow: `pip install tensorflow`. "
        "The TensorFlow package is required for data preprocessing with any backend."
    )

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.tokenizers import tokenizer
from keras_nlp.src.utils.tensor_utils import assert_tf_text_installed
from keras_nlp.src.utils.tensor_utils import convert_to_ragged_batch
from keras_nlp.src.utils.tensor_utils import is_int_dtype
from keras_nlp.src.utils.tensor_utils import is_string_dtype

try:
    import tensorflow_text as tf_text
except ImportError:
    tf_text = None

VOCAB_FILENAME = "vocabulary.json"
MERGES_FILENAME = "merges.txt"


# As python and TF handles special spaces differently, we need to
# manually handle special spaces during string split.
SPECIAL_WHITESPACES = r"\x{a0}\x{2009}\x{202f}\x{3000}"

# String splitting regex pattern.
SPLIT_PATTERN_1 = (
    r"'s|'t|'re|'ve|'m|'ll|'d"
    + r"|[\s{special_spaces}]+[\n\r\t\f६{special_spaces}]| ?\p{L}+|"
    + r" ?[\p{N}]+| ?[^\s\p{L}\p{N}{special_spaces}]+"
)
SPLIT_PATTERN_1 = SPLIT_PATTERN_1.replace(
    "{special_spaces}", SPECIAL_WHITESPACES
)
SPLIT_PATTERN_2 = rf"""[\s६{SPECIAL_WHITESPACES}]$"""


def create_alts_for_unsplittable_tokens(unsplittable_tokens):
    # Create alternates for all special tokens that will be not split during
    # tokenization.
    alts = []
    prefix = "Ĵ"
    # Trim out splitters.
    replace_pattern = r"'|\s+|[^\p{L}\p{N}]+"
    for token in unsplittable_tokens:
        token = re.sub(replace_pattern, "", token)
        alts.append(prefix + token)
    return alts


def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    # removes mapping an int to a whitespace character
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    bs = [n.to_bytes(1, "little") for n in bs]
    return bs, cs  # int to string mapping


def remove_strings_from_inputs(tensor, string_to_remove):
    """Remove certain strings from input tensor."""
    non_empty_mask = tensor != string_to_remove
    flatten_indexes = tf.where(non_empty_mask)
    flatten_result = tf.gather_nd(tensor, flatten_indexes)
    row_lengths = tf.reduce_sum(tf.cast(non_empty_mask, "int64"), axis=1)
    result = tf.RaggedTensor.from_row_lengths(
        values=flatten_result,
        row_lengths=row_lengths,
    )
    return result


def split_strings_for_bpe(inputs, unsplittable_tokens=None):
    # We need to recreate the exact behavior of token presplitting in the
    # original gpt2 tokenizer which uses a lookahead. As re2 does not
    # support lookahead match, we are using an alternative insert a special
    # token "६" before leading space of non-space characters and after the
    # trailing space, e.g., " keras" will be "६ keras".
    inputs = tf.strings.regex_replace(
        inputs, rf"( )([^\s{SPECIAL_WHITESPACES}])", r"६\1\2"
    )
    inputs = tf.strings.regex_replace(
        inputs, rf"(\s{SPECIAL_WHITESPACES})$", r"\1६"
    )
    if unsplittable_tokens:
        alts = create_alts_for_unsplittable_tokens(unsplittable_tokens)
        for token, alt in zip(unsplittable_tokens, alts):
            escaped_token = re.escape(token)
            inputs = tf_text.regex_split(inputs, escaped_token, escaped_token)
            inputs = tf.strings.regex_replace(inputs, escaped_token, alt)
    raw_tokens = tf_text.regex_split(inputs, SPLIT_PATTERN_1, SPLIT_PATTERN_1)
    # Second pass splits out the last whilespace char or "६".
    raw_tokens = tf_text.regex_split(
        raw_tokens, SPLIT_PATTERN_2, SPLIT_PATTERN_2
    )
    if unsplittable_tokens:
        # Replace special tokens alternate with originals.
        for token, alt in zip(unsplittable_tokens, alts):
            escaped_alt = re.escape(alt)
            raw_tokens = tf.strings.regex_replace(
                raw_tokens, escaped_alt, token
            )
    while raw_tokens.shape.rank > 2:
        raw_tokens = raw_tokens.merge_dims(1, 2)
    return remove_strings_from_inputs(raw_tokens, "६")


class BytePairTokenizerCache(tf.Module):
    """Cache that stores the encoded result of seen tokens.

    The cache key is string tensor or python strings, and the value is split
    tokens joined by whitespace. For example, "dragonfly" => "dragon fly"

    Example:
    ```
    cache = BytePairTokenizerCache()
    cache.insert(["butterfly", "dragonfly"], ["but ter fly", "dragon fly"])
    cache.lookup(["butterfly"])
    ```
    """

    def __init__(self):
        # `tf.lookup.experimental.MutableHashTable` does not support string to
        # string mapping. So we first convert to string to an integer key, and
        # use the integer key to find the value.
        self.factors = tf.pow(
            tf.constant(256, dtype="int64"), tf.range(0, 8, dtype="int64")
        )
        self.id2value = tf.lookup.experimental.MutableHashTable(
            "int64", tf.string, ""
        )

    def _get_key(self, keys):
        """Get the hash key for given inputs."""
        # `tf.fingerprint` converts token to a array of uint8 of length 8, we
        # need to convert it to a uint64.
        return tf.squeeze(
            tf.matmul(
                tf.cast(tf.fingerprint(keys), dtype="int64"),
                self.factors[:, tf.newaxis],
            ),
            -1,
        )

    def lookup(self, keys):
        """Look up the encoded outputs of given tokens."""
        ids = self._get_key(keys)
        result = self.id2value.lookup(ids)
        # Ensure output shape for graph mode.
        result.set_shape([None])
        return result

    def insert(self, keys, values):
        """Insert token <=> encoded outputs pairs."""
        self.id2value.insert(self._get_key(keys), values)


def create_static_hashtable(keys, values, default):
    return tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tf.convert_to_tensor(keys),
            tf.convert_to_tensor(values),
        ),
        default_value=default,
    )


@keras_nlp_export("keras_nlp.tokenizers.BytePairTokenizer")
class BytePairTokenizer(tokenizer.Tokenizer):
    """Bype-pair encoding tokenizer layer.

    This BPE tokenizer provides the same functionality as the official GPT-2
    tokenizer. Given the same `vocabulary` which maps tokens to ids, and `merges`
    which describes BPE merge rules, it should provide the same output
    as OpenAI implementation (https://github.com/openai/gpt-2/blob/master/src/encoder.py).
    Different from OpenAI, this implementation is graph-compatible, so you can
    use it within a `tf.data` pipeline.

    If input is a batch of strings (rank > 0):
    By default, the layer will output a `tf.RaggedTensor` where the last
    dimension of the output is ragged. If `sequence_length` is set, the layer
    will output a dense `tf.Tensor` where all inputs have been padded or
    truncated to `sequence_length`.
    If input is a scalar string (rank == 0):
    By default, the layer will output a dense `tf.Tensor` with static shape
    `[None]`. If `sequence_length` is set, the output will be
    a dense `tf.Tensor` of shape `[sequence_length]`.

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
    >>> tokenizer = keras_nlp.tokenizers.BytePairTokenizer(vocab, merge)
    >>> outputs = tokenizer("butterfly")
    >>> np.array(outputs)
    array([1, 2], dtype=int32)
    >>> seq1, seq2 = tokenizer(["butterfly", "butter"])
    >>> np.array(seq1)
    array([1, 2], dtype=int32)
    >>> np.array(seq2)
    array([1], dtype=int32)
    >>> tokenizer = keras_nlp.tokenizers.BytePairTokenizer(
    ...     vocab, merge, sequence_length=2)
    >>> seq1, seq2 = tokenizer(["butterfly", "butter"])
    >>> np.array(seq1)
    array([1, 2], dtype=int32)
    >>> np.array(seq2)
    array([1, 0], dtype=int32)

    Detokenize
    >>> vocab = {"butter": 1, "fly": 2}
    >>> merge = ["b u", "t t", "e r", "bu tt", "butt er", "f l", "fl y"]
    >>> tokenizer = keras_nlp.tokenizers.BytePairTokenizer(vocab, merge)
    >>> tokenizer.detokenize([[1, 2]])
    <tf.Tensor: shape=(1,), dtype=string, numpy=array([b'butterfly'],
    dtype=object)>
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
        assert_tf_text_installed(self.__class__.__name__)

        if not is_int_dtype(dtype) and not is_string_dtype(dtype):
            raise ValueError(
                "Output dtype must be an integer type or a string. "
                f"Received: dtype={dtype}"
            )

        super().__init__(dtype=dtype, **kwargs)
        self.sequence_length = sequence_length
        self.add_prefix_space = add_prefix_space
        self.unsplittable_tokens = unsplittable_tokens
        self.file_assets = [VOCAB_FILENAME, MERGES_FILENAME]

        # Create byte <=> unicode mapping. This is useful for handling
        # whitespace tokens.
        byte_list, unicode_list = bytes_to_unicode()
        self.byte2unicode = create_static_hashtable(
            byte_list, unicode_list, default=""
        )
        self.unicode2byte = create_static_hashtable(
            unicode_list, byte_list, default=""
        )

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
            self.cache = None
            self.id_to_token_map = None
            self.token_to_id_map = None
            self.merge_ranks_lookup_default = None
            self.merge_ranks = None
            return

        if isinstance(vocabulary, str):
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
            with open(merges, encoding="utf-8") as f:
                self.merges = [bp.rstrip() for bp in f]
        elif isinstance(merges, Iterable):
            self.merges = list(merges)
        else:
            raise ValueError(
                "Merges must be a file path or a list of merge rules. "
                f"Received: `type(merges)={type(merges)}`"
            )

        self.cache = BytePairTokenizerCache()
        if self.unsplittable_tokens:
            # Put special tokens into cache, so it won't be further split and
            # merged.
            self.cache.insert(
                self.unsplittable_tokens, self.unsplittable_tokens
            )

        # Create mapping between string tokens to int ids, and vice versa.
        byte_pairs = [x[0] for x in self.vocabulary.items()]
        byte_pair_encoding_indices = [x[1] for x in self.vocabulary.items()]
        self.token_to_id_map = create_static_hashtable(
            byte_pairs,
            byte_pair_encoding_indices,
            default=-1,
        )
        self.id_to_token_map = create_static_hashtable(
            byte_pair_encoding_indices,
            byte_pairs,
            default="",
        )

        # Create ranking of merge rules, this is the same as order of merge
        # pairs in `self.merges`.
        self.merge_ranks_lookup_default = len(self.merges) + 1
        self.merge_ranks = create_static_hashtable(
            self.merges,
            list(range(len(self.merges))),
            default=self.merge_ranks_lookup_default,
        )

    def get_vocabulary(self):
        """Get the tokenizer vocabulary as a list of strings tokens."""
        self._check_vocabulary()
        return self.vocabulary.keys()

    def vocabulary_size(self):
        """Get the integer size of the tokenizer vocabulary."""
        self._check_vocabulary()
        return len(self.vocabulary)

    def id_to_token(self, id):
        """Convert an integer id to a string token."""
        # This will be slow, but keep memory usage down compared to building a
        # dict. Assuming the main use case is looking up a few special tokens
        # early in the vocab, this should be fine.
        self._check_vocabulary()

        keys = self.get_vocabulary()
        for token in keys:
            if self.vocabulary[token] == id:
                return token
        raise ValueError(f"`id` is out of the vocabulary. Received: {id}")

    def token_to_id(self, token):
        """Convert a string token to an integer id."""
        self._check_vocabulary()
        return self.vocabulary[token]

    @tf.function
    def _bpe_merge_one_step(self, words, mask):
        """Perform one step of byte-pair merge."""
        # Get all word pairs.
        first, second = words[:, :-1], words[:, 1:]

        # Mask empty.
        non_empty_mask = second.nested_row_lengths()[0] != 0
        mask = mask & non_empty_mask
        if not tf.reduce_any(mask):
            return [words, mask]
        non_empty_indices = tf.boolean_mask(tf.range(tf.shape(mask)[0]), mask)
        filterd_first = tf.ragged.boolean_mask(first, mask)
        filtered_second = tf.ragged.boolean_mask(second, mask)

        # Get byte pair ranking in merge rules.
        pairs = tf.strings.join([filterd_first, filtered_second], separator=" ")
        pair_rank = self.merge_ranks.lookup(pairs)

        # Get BPE pair ranks.
        min_pair_rank = tf.reduce_min(pair_rank, axis=1)
        pair_found_mask = min_pair_rank != self.merge_ranks_lookup_default

        # Tokens that cannot be further merged are marked as finished.
        mask = tf.tensor_scatter_nd_update(
            mask, tf.expand_dims(non_empty_indices, axis=1), pair_found_mask
        )
        if not tf.math.reduce_any(mask):
            return [words, mask]

        masked_pair_rank = tf.ragged.boolean_mask(pair_rank, pair_found_mask)
        min_pair_rank_indices = tf.math.argmin(
            masked_pair_rank.to_tensor(self.merge_ranks_lookup_default), axis=1
        )

        # Get words and pairs to process.
        unfinished_words = tf.ragged.boolean_mask(words, mask)

        pair_left = tf.gather(
            unfinished_words, min_pair_rank_indices, batch_dims=1
        )
        pair_right = tf.gather(
            unfinished_words, min_pair_rank_indices + 1, batch_dims=1
        )

        merged_pairs = tf.strings.join([pair_left, pair_right])
        empty_strs = tf.fill(tf.shape(merged_pairs), "")

        unfinished_word_indices = tf.cast(
            tf.boolean_mask(tf.range(tf.shape(mask)[0]), mask), dtype="int64"
        )
        merged_pair_indices = tf.concat(
            [
                unfinished_word_indices[:, tf.newaxis],
                min_pair_rank_indices[:, tf.newaxis],
            ],
            axis=1,
        )
        empty_string_indices = tf.concat(
            [
                unfinished_word_indices[:, tf.newaxis],
                min_pair_rank_indices[:, tf.newaxis] + 1,
            ],
            axis=1,
        )

        tensor_words = words.to_tensor(default_value="")
        tensor_words = tf.tensor_scatter_nd_update(
            tensor_words,
            merged_pair_indices,
            merged_pairs,
        )

        words = tf.tensor_scatter_nd_update(
            tensor_words,
            empty_string_indices,
            empty_strs,
        )
        # Remove empty strings.
        words = remove_strings_from_inputs(words, "")
        return [words, mask]

    def _bpe_merge(self, inputs):
        """Perform byte-pair merge for each word in the inputs."""
        num_words = tf.shape(inputs)[0]

        # Merge bytes.
        def loop_condition(_, mask):
            return tf.math.reduce_any(mask)

        initial_mask = tf.fill((num_words,), True)
        merged_words, _ = tf.while_loop(
            loop_condition,
            self._bpe_merge_one_step,
            loop_vars=[
                inputs,
                initial_mask,
            ],
            shape_invariants=[
                tf.TensorShape([None, None]),
                tf.TensorShape([None]),
            ],
        )
        return merged_words

    def _check_vocabulary(self):
        if self.vocabulary is None:
            raise ValueError(
                "No vocabulary has been set for BytePairTokenizer. Make sure "
                "to pass `vocabulary` and `merges` arguments when creating the "
                "layer."
            )

    def tokenize(self, inputs):
        self._check_vocabulary()
        if not isinstance(inputs, (tf.Tensor, tf.RaggedTensor)):
            inputs = tf.convert_to_tensor(inputs)

        if self.add_prefix_space:
            inputs = tf.strings.join([" ", inputs])

        scalar_input = inputs.shape.rank == 0
        if scalar_input:
            inputs = tf.expand_dims(inputs, 0)

        raw_tokens = split_strings_for_bpe(inputs, self.unsplittable_tokens)
        token_row_splits = raw_tokens.row_splits
        flat_tokens = raw_tokens.flat_values

        # Check cache.
        cache_lookup = self.cache.lookup(flat_tokens)
        cache_mask = cache_lookup == ""

        has_unseen_words = tf.math.reduce_any(
            (cache_lookup == "") & (flat_tokens != "")
        )

        def process_unseen_tokens():
            unseen_tokens = tf.boolean_mask(flat_tokens, cache_mask)
            self._bpe_merge_and_update_cache(unseen_tokens)
            return self.cache.lookup(flat_tokens)

        # If `has_unseen_words == True`, it means not all tokens are in cache,
        # we will process the unseen tokens. Otherwise return the cache lookup.
        tokenized_words = tf.cond(
            has_unseen_words,
            process_unseen_tokens,
            lambda: cache_lookup,
        )

        tokens = tf.strings.split(tokenized_words, sep=" ")
        if self.compute_dtype != tf.string:
            # Encode merged tokens.
            tokens = self.token_to_id_map.lookup(tokens)

        # Unflatten to match input.
        tokens = tf.RaggedTensor.from_row_splits(
            tokens.flat_values,
            tf.gather(tokens.row_splits, token_row_splits),
        )

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
        self._check_vocabulary()
        inputs, unbatched, _ = convert_to_ragged_batch(inputs)
        inputs = tf.cast(inputs, self.dtype)
        unicode_text = tf.strings.reduce_join(
            self.id_to_token_map.lookup(inputs), axis=-1
        )
        split_unicode_text = tf.strings.unicode_split(unicode_text, "UTF-8")
        outputs = tf.strings.reduce_join(
            self.unicode2byte.lookup(split_unicode_text), axis=-1
        )

        if unbatched:
            outputs = tf.squeeze(outputs, 0)
        return outputs

    def compute_output_spec(self, input_spec):
        return keras.KerasTensor(
            input_spec.shape + (self.sequence_length,), dtype=self.compute_dtype
        )

    def _transform_bytes(self, tokens):
        """Map token bytes to unicode using `byte2unicode`."""
        split_bytes = tf.strings.bytes_split(tokens)
        split_unicode = self.byte2unicode.lookup(split_bytes)
        return split_unicode

    def _bpe_merge_and_update_cache(self, tokens):
        """Process unseen tokens and add to cache."""
        words = self._transform_bytes(tokens)
        tokenized_words = self._bpe_merge(words)

        # For each word, join all its token by a whitespace,
        # e.g., ["dragon", "fly"] => "dragon fly" for hash purpose.
        tokenized_words = tf.strings.reduce_join(
            tokenized_words, axis=1, separator=" "
        )
        self.cache.insert(tokens, tokenized_words)

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
