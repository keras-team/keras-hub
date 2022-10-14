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

import json
from typing import Iterable
from typing import List

import tensorflow as tf
import tensorflow_text as tf_text

from keras_nlp.tokenizers import tokenizer


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


class BytePairTokenizerCache:
    """Cache that stores the encoded result of seen tokens."""

    def __init__(self):
        # `tf.lookup.experimental.MutableHashTable` does not support string to
        # string mapping. So we first convert to string to an integer key, and
        # use the integer key to find the value.
        self.factors = tf.pow(256, tf.range(0, 8, dtype=tf.int64))
        self.id2value = tf.lookup.experimental.MutableHashTable(
            tf.int64, tf.string, ""
        )

    def get_key(self, keys):
        """Get the hash key for given inputs."""
        # `tf.fingerprint` converts token to a array of uint8 of length 8, we
        # need to convert it to a uint64.
        return tf.squeeze(
            tf.matmul(
                tf.cast(tf.fingerprint(keys), dtype=tf.int64),
                self.factors[:, tf.newaxis],
            ),
            -1,
        )

    def lookup(self, keys):
        """Look up the encoded outputs of given tokens."""
        ids = self.get_key(keys)
        result = self.id2value.lookup(ids)
        # Ensure output shape for graph mode.
        result.set_shape([None])
        return result

    def insert(self, keys, values):
        """Insert token <=> encoded outputs pairs."""
        self.id2value.insert(self.get_key(keys), values)


def create_static_hashtable(keys, values, default):
    hashtable = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tf.convert_to_tensor(keys),
            tf.convert_to_tensor(values),
        ),
        default_value=default,
    )
    return hashtable


class BytePairTokenizer(tokenizer.Tokenizer):
    """Bype-pair encoder.

    This BPE encoder provides the same funtionality as official GPT2 tokenizer.
    Given the same `vocabulary` and `merges`, it should provide the same output
    as fairseq implementation (https://github.com/facebookresearch/fairseq/blob/main/fairseq/data/encoders/gpt2_bpe.py).
    Different from fairseq, this implementation is graph-compatible, so you can
    use it within a tf.data pipeline.

    Args:
        vocabulary: string or dict, maps token to integer ids. If it is a
            string, it should be the file path to a json file.
        merges: string or list, contains the merge rule. If it is a string,
            it should be the file path to merge rules. The merge rule file
            should have one merge rule per line.
        sequence_length: int, defaults to None. If set, the output will be
            padded or truncated to the `sequence_length`.
    """

    def __init__(
        self,
        vocabulary,
        merges,
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

        if isinstance(vocabulary, str):
            with open(vocabulary, "r") as f:
                self.vocabulary = json.load(f)
        elif isinstance(vocabulary, dict):
            self.vocabulary = vocabulary.copy()
        else:
            raise ValueError(
                "Vocabulary must be an file path or dictionary mapping string "
                f"token to int ids. Received type: {type(vocabulary)}."
            )
        if isinstance(merges, str):
            self.merges = [bp.rstrip() for bp in tf.io.gfile.GFile(merges)]
        elif isinstance(merges, Iterable):
            self.merges = list(merges)
        else:
            raise ValueError(
                "Merges must be a file path or a list of merge rules. "
                f"Received type: {type(merges)}."
            )
        self.sequence_length = sequence_length

        # String splitting regex pattern.
        self.special_space = r"\x{a0}\x{2009}\x{202f}\x{3000}"

        self.pat1 = r"""'s|'t|'re|'ve|'m|'ll|'d
            |[\s{special_space}]+[\n\r\t\f६{special_space}]| ?\p{L}+| ?[\p{N}]+
            | ?[^\s\p{L}\p{N}{special_space}]+"""
        self.pat1 = self.pat1.replace("{special_space}", self.special_space)
        self.pat2 = rf"""[\s६{self.special_space}]$"""

        # Create byte <=> unicode mapping. This is useful for handling
        # whitespace tokens.
        byte_list, unicode_list = bytes_to_unicode()
        self.byte2unicode = create_static_hashtable(
            byte_list, unicode_list, default=""
        )
        self.unicode2byte = create_static_hashtable(
            unicode_list, byte_list, default=""
        )

        self.cache = BytePairTokenizerCache()

        # Create mapping between string tokens to int ids, and vice versa.
        byte_pairs = [x[0] for x in self.vocabulary.items()]
        byte_pair_encoding_idxs = [x[1] for x in self.vocabulary.items()]
        self.token_to_id_map = create_static_hashtable(
            byte_pairs,
            byte_pair_encoding_idxs,
            default=-1,
        )
        self.id_to_token_map = create_static_hashtable(
            byte_pair_encoding_idxs,
            byte_pairs,
            default="",
        )

        # Create ranking of merge rules, this is the same as order of merge
        # pairs in `self.merges`.
        self.max_bpe_rank = len(self.merges) + 1
        self.bpe_ranks = create_static_hashtable(
            self.merges,
            list(range(len(self.merges))),
            default=self.max_bpe_rank,
        )

    def get_vocabulary(self) -> List[str]:
        """Get the tokenizer vocabulary as a list of strings tokens."""
        return self.vocabulary.keys()

    def vocabulary_size(self) -> int:
        """Get the size of the tokenizer vocabulary."""
        return len(self.vocabulary)

    def id_to_token(self, id: int) -> str:
        """Convert an integer id to a string token."""
        # This will be slow, but keep memory usage down compared to building a
        # . Assuming the main use case is looking up a few special tokens
        # early in the vocab, this should be fine.
        for token in range(self.vocabulary):
            if self.vocabulary[token] == id:
                return token
        return None

    def token_to_id(self, token: str) -> int:
        """Convert a string token to an integer id."""
        return self.vocabulary[token]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                # Ideally a vocabulary would be saved as a plain text asset in
                # the saved model. We have no good way to support this
                # currently, so we save the vocabulary in the config.
                "vocabulary": self.vocabulary,
                "merges": self.merges,
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

        # As re2 does not support lookahead match, we are using an alternative
        # to insert a special token "६" before leading space of non-space
        # characters and after the trailing space, e.g., " keras" will be
        # "६ keras".
        inputs = tf.strings.regex_replace(
            inputs, rf"( )([^\s{self.special_space}])", r"६\1\2"
        )
        inputs = tf.strings.regex_replace(
            inputs, rf"(\s{self.special_space})$", r"\1६"
        )
        raw_tokens = tf_text.regex_split(inputs, self.pat1, self.pat1)
        # Second pass splits out the last whilespace char or "६".
        raw_tokens = tf_text.regex_split(raw_tokens, self.pat2, self.pat2)
        if raw_tokens.shape.rank > 2:
            raw_tokens = raw_tokens.merge_dims(1, 2)
        raw_tokens = self._remove_whitespace_placeholder(raw_tokens)
        token_row_splits = raw_tokens.row_splits
        flatten_tokens = raw_tokens.flat_values

        # Check cache.
        cache_lookup = self.cache.lookup(flatten_tokens)
        cache_mask = cache_lookup == ""

        if (
            tf.math.count_nonzero(
                tf.boolean_mask(cache_mask, flatten_tokens != "")
            )
            == 0
        ):
            # All elements are in cache.
            result = cache_lookup
        else:
            # Create byte pair merges and add to cache.
            unseen_tokens = tf.boolean_mask(flatten_tokens, cache_mask)
            self._byte_pair_encoding(unseen_tokens)
            result = self.cache.lookup(flatten_tokens)

        # Encode merged tokens.
        result = tf.strings.split(result, sep=" ")
        encoding = self.token_to_id_map.lookup(result)

        # Unflatten to match input.
        encoding = tf.RaggedTensor.from_row_splits(
            encoding.flat_values,
            tf.gather(encoding.row_splits, token_row_splits),
        )

        # Convert to a dense output if `sequence_length` is set.
        if self.sequence_length:
            output_shape = encoding.shape.as_list()
            output_shape[-1] = self.sequence_length
            encoding = encoding.to_tensor(shape=output_shape)
        # Convert to a dense output if input in scalar
        if scalar_input:
            encoding = tf.squeeze(encoding, 0)
            tf.ensure_shape(encoding, shape=[self.sequence_length])

        return encoding

    def detokenize(self, inputs):
        if not isinstance(inputs, (tf.Tensor, tf.RaggedTensor)):
            inputs = tf.convert_to_tensor(inputs)

        scalar_input = inputs.shape.rank == 0
        if scalar_input:
            inputs = tf.expand_dims(inputs, 0)

        unicode_text = tf.strings.reduce_join(
            self.id_to_token_map.lookup(inputs), axis=1
        )
        split_unicode_text = tf.strings.unicode_split(unicode_text, "UTF-8")
        byte_text = tf.strings.reduce_join(
            self.unicode2byte.lookup(split_unicode_text)
        )

        if not scalar_input:
            byte_text = tf.expand_dims(byte_text, 0)

        return byte_text

    def _encode_tokens(self, tokens):
        """Map token bytes to unicode using `byte2unicode`."""
        split_bytes = tf.strings.bytes_split(tokens)
        split_unicode = self.byte2unicode.lookup(split_bytes)
        return split_unicode

    def _remove_strings(self, tensor, string_to_remove):
        """Remove certain strings from input tensor."""
        non_empty_mask = tensor != string_to_remove
        flatten_indexes = tf.where(non_empty_mask)
        flatten_result = tf.gather_nd(tensor, flatten_indexes)
        row_lengths = tf.reduce_sum(tf.cast(non_empty_mask, tf.int64), axis=1)
        result = tf.RaggedTensor.from_row_lengths(
            values=flatten_result,
            row_lengths=row_lengths,
        )
        return result

    def _remove_empty_strings(self, tensor):
        """Remove empty strings in a tensor"""
        return self._remove_strings(tensor, "")

    def _remove_whitespace_placeholder(self, tensor):
        return self._remove_strings(tensor, "६")

    @tf.function
    def _byte_pair_merge_loop_body(self, words, mask):
        """Iterative merging process for byte pair encoding algorithm.

        The end condition is either the word has been fully merged (list has
        only one byte string), or it can no longer perform a merge.
        """
        # Get all word pairs.
        first, second = words[:, :-1], words[:, 1:]

        # Mask empty.
        non_empty_mask = second.nested_row_lengths()[0] != 0
        mask = tf.logical_and(mask, non_empty_mask)
        if tf.math.count_nonzero(mask) == 0:
            return [words, mask]
        non_empty_idxs = tf.boolean_mask(tf.range(tf.shape(mask)[0]), mask)
        tmp_first = tf.ragged.boolean_mask(first, mask)
        tmp_second = tf.ragged.boolean_mask(second, mask)

        # Get byte pair ranking in merge rules.
        pairs = tf.strings.join([tmp_first, tmp_second], separator=" ")
        pair_rank = self.bpe_ranks.lookup(pairs)

        # Get BPE pair ranks.
        min_pair_rank = tf.reduce_min(pair_rank, axis=1)
        not_found_mask = min_pair_rank != self.max_bpe_rank

        # Tokens cannot be further merged are marked as finished.
        mask = tf.tensor_scatter_nd_update(
            mask, tf.expand_dims(non_empty_idxs, axis=1), not_found_mask
        )
        if tf.math.count_nonzero(mask) == 0:
            return [words, mask]

        masked_pair_rank = tf.ragged.boolean_mask(pair_rank, not_found_mask)
        min_pair_rank_idx = tf.math.argmin(
            masked_pair_rank.to_tensor(self.max_bpe_rank), axis=1
        )

        # Get words and pairs to process.
        unfinished_words = tf.ragged.boolean_mask(words, mask)

        pair_left = tf.gather(unfinished_words, min_pair_rank_idx, batch_dims=1)
        pair_right = tf.gather(
            unfinished_words, min_pair_rank_idx + 1, batch_dims=1
        )

        merged_pairs = tf.strings.join([pair_left, pair_right])
        empty_strs = tf.fill(tf.shape(merged_pairs), "")

        unfinished_indices = tf.cast(
            tf.boolean_mask(tf.range(tf.shape(mask)[0]), mask), dtype=tf.int64
        )
        merge_update_indices_left = tf.concat(
            [
                unfinished_indices[:, tf.newaxis],
                min_pair_rank_idx[:, tf.newaxis],
            ],
            axis=1,
        )
        merge_update_indices_right = tf.concat(
            [
                unfinished_indices[:, tf.newaxis],
                min_pair_rank_idx[:, tf.newaxis] + 1,
            ],
            axis=1,
        )

        tensor_words = words.to_tensor(default_value="")
        tensor_words = tf.tensor_scatter_nd_update(
            tensor_words,
            merge_update_indices_left,
            merged_pairs,
        )

        words = tf.tensor_scatter_nd_update(
            tensor_words,
            merge_update_indices_right,
            empty_strs,
        )
        words = self._remove_empty_strings(words)
        return [words, mask]

    def _byte_pair_encoding(self, tokens):
        """Process unseen tokens and add to cache."""
        words = self._encode_tokens(tokens)
        if isinstance(words, tf.RaggedTensor):
            num_words = words.bounding_shape(0)
        else:
            num_words = tf.shape(words)[0]

        # Merge bytes.
        def loop_condition(words, mask):
            return tf.math.count_nonzero(mask) > 0

        initial_mask = tf.fill((num_words,), True)
        merged_words, _ = tf.while_loop(
            loop_condition,
            self._byte_pair_merge_loop_body,
            loop_vars=[words, initial_mask],
            shape_invariants=[
                tf.TensorShape([None, None]),
                tf.TensorShape([None]),
            ],
        )

        merged_words_hash = tf.strings.reduce_join(
            merged_words, axis=1, separator=" "
        )
        self.cache.insert(tokens, merged_words_hash)
