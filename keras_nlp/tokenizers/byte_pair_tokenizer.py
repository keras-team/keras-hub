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

from lib2to3.pgen2 import token
from typing import Dict
from typing import List
from typing import Iterable
from venv import create

import tensorflow as tf
import tensorflow_text as tf_text
import json

from keras_nlp.tokenizers import tokenizer

def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    #removes mapping an int to a whitespace character
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return bs, cs #int to string mapping

class BytePairTokenizerCache():
    def __init__(self):
        self.key2id = tf.lookup.experimental.DenseHashTable(
            tf.string, tf.int64, -1, "a ", "b "
        )
        self.id2value = tf.lookup.experimental.MutableHashTable(
            tf.int64, tf.string, ""
        )
        self.id = tf.Variable(0, dtype=tf.int64)
    def lookup(self, keys):
        """Look up a tensor of tokens."""
        ids = self.key2id.lookup(keys)
        result = self.id2value.lookup(ids)
        # Ensure output shape for graph mode.
        result.set_shape([None])
        return result

    def insert(self, keys, values):
        """Insert a tensor of tokens to bp words mapping"""
        size = tf.cast(tf.shape(keys)[0], tf.int64)
        ids = tf.range(self.id, self.id+size)
        self.id.assign(self.id+size)

        self.key2id.insert(keys, ids)
        self.id2value.insert(ids, values)
        return ids

def create_static_hashtable(keys, values, default):
    hashtable = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(
          tf.convert_to_tensor(keys), 
          tf.convert_to_tensor(values),
      ),
      default_value=default
    )
    return hashtable


class BytePairTokenizer(tokenizer.Tokenizer):
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
            # Make a copy.
            self.vocabulary = vocabulary.copy()
        else:
            raise ValueError(
                "Vocabulary must be an file path or dictionary mapping byte "
                f"pairs to token ids. Received: vocabulary={vocabulary}."
            )
        if isinstance(merges, str):
            self.merges = [
                bp.rstrip() for bp in tf.io.gfile.GFile(merges)
            ]
        elif isinstance(merges, Iterable):
            self.merges = list(merges)
        else:
            raise ValueError(
                "Merges must be a file path or a list of merges. Recieved: "
                f"merges={merges}."
            )
        self.sequence_length = sequence_length

        # TODO: use dtype to cast output
        self.pat = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+"""

        # Map byte to unicode.
        bs, cs = bytes_to_unicode()
        self.byte2unicode = create_static_hashtable(bs, cs, default='')

        # Caching.
        self.cache = BytePairTokenizerCache()

        # BytePair encodings.
        self.byte_pair_encoder = create_static_hashtable(
            [x[0] for x in self.vocabulary.items()],
            [x[1] for x in self.vocabulary.items()],
            default=-1
        )

        # Merging rankings.
        self.max_bpe_rank = len(self.merges)+1
        self.bpe_ranks = create_static_hashtable(
            self.merges,
            list(range(len(self.merges))),
            default=self.max_bpe_rank
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

        # Regex match tokens.
        raw_tokens = tf_text.regex_split(inputs, self.pat, self.pat)
        token_row_splits = raw_tokens.row_splits
        flatten_tokens = raw_tokens.flat_values

        # Check cache.
        cache_lookup = self.cache.lookup(flatten_tokens)
        cache_mask = cache_lookup == ""
        
        if tf.math.count_nonzero(tf.boolean_mask(cache_mask, flatten_tokens != "")) == 0:
            # All elements are in cache.
            result = cache_lookup
        else:
            # Create byte pair merges and add to cache.
            unseen_tokens = tf.boolean_mask(flatten_tokens, cache_mask)
            self._byte_pair_encoding(unseen_tokens)
            result = self.cache.lookup(flatten_tokens)
        
        # Encode merged tokens.
        result = tf.strings.split(result, sep=" ")
        encoding = self.byte_pair_encoder.lookup(result)

        # Unflatten to match input.
        encoding = tf.RaggedTensor.from_row_splits(
            encoding.flat_values, 
            tf.gather(
                encoding.row_splits, 
                token_row_splits
            )
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

    # Helper functions go here.

    def _encode_tokens(self, tokens):
        """Map token bytes to unicode using `byte2unicode`."""
        #TODO: This could be optimized.
        # Encode token bytes.
        token_bytes = tf.strings.bytes_split(tokens)
        flatten_bytes = token_bytes.flat_values
        flatten_bytes = tf.squeeze(
            tf.cast(
                tf.io.decode_raw(flatten_bytes, tf.uint8), tf.int32
            )
        )
        flatten_unicode = self.byte2unicode.lookup(flatten_bytes)
        token_unicode = tf.RaggedTensor.from_row_lengths(
            values=flatten_unicode,
            row_lengths=token_bytes.row_lengths()
        )
        return token_unicode

    def _remove_empty_strings(self, tensor):
        """Remove empty strings in a tensor"""
        non_empty_mask = tensor != ""
        flatten_indexes = tf.where(non_empty_mask)
        flatten_result = tf.gather_nd(tensor, flatten_indexes)
        row_lengths = tf.reduce_sum(tf.cast(non_empty_mask, tf.int64), axis=1)
        result = tf.RaggedTensor.from_row_lengths(
            values=flatten_result,
            row_lengths=row_lengths,
        )
        return result

    def _find_top_pair_and_merge(self, words, top_pair_first, top_pair_second):
        """Merges the top pair in word."""
        # Get shifted word tokens.
        word_pair_first = words[:, :-1]
        word_pair_second = words[:, 1:]

        # Get top pair occurances.
        top_pair_first = tf.expand_dims(top_pair_first, axis=1)
        top_pair_second = tf.expand_dims(top_pair_second, axis=1)
        top_pair_starts = tf.math.logical_and(
            word_pair_first==top_pair_first, 
            word_pair_second==top_pair_second
        )
        
        # Fixing off by one indexing.
        num_words = tf.shape(top_pair_starts)[0]
        front_mask = tf.logical_not(
            tf.concat(
                [tf.fill([num_words, 1], False), top_pair_starts], 1
            )
        )
        back_mask = tf.concat(
            [tf.fill([num_words, 1], False), top_pair_starts], 1
        )

        # Filter word tokens to keep.
        front = tf.where(front_mask, words, "")
        # Filter `top_pair_second` tokens to merge.
        back = tf.concat(
            [tf.where(back_mask[:, 1:], word_pair_second, ""), tf.fill([num_words, 1], "")], 1
        )
        # Merge and clean up empty strings.
        joined = tf.strings.join([front, back])
        return self._remove_empty_strings(joined)

    def _get_pairs(self, words):
        return words[:, :-1], words[:, 1:]

    @tf.function
    def _byte_pair_merge_loop_body(self, words, mask):
        """Iterative merging process for byte pair encoding algorithm."""
        # Get all word pairs.
        first, second = self._get_pairs(words)
        
        # Mask empty.
        non_empty_mask = second.nested_row_lengths()[0] != 0
        mask = tf.logical_and(mask, non_empty_mask)
        if tf.math.count_nonzero(mask) == 0:
            return [words, mask]
        non_empty_idxs = tf.boolean_mask(tf.range(tf.shape(mask)[0]), mask)
        tmp_first = tf.ragged.boolean_mask(first, mask)
        tmp_second = tf.ragged.boolean_mask(second, mask)

        # Get top word pair.
        pair_hash = tf.strings.join([tmp_first, tmp_second], separator=" ")
        pair_rank = self.bpe_ranks.lookup(pair_hash)

        # Get BPE pair ranks.
        min_pair_rank = tf.reduce_min(pair_rank, axis=1)
        not_found_mask = min_pair_rank != self.max_bpe_rank
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
        p_words = tf.ragged.boolean_mask(words, mask)
        p_first = tf.ragged.boolean_mask(first, mask)
        p_second = tf.ragged.boolean_mask(second, mask)
        p_min_rank_first = tf.gather(p_first, min_pair_rank_idx, batch_dims=1)
        p_min_rank_second = tf.gather(p_second, min_pair_rank_idx, batch_dims=1)

        # Process merges of top pairs.
        p_words = self._find_top_pair_and_merge(p_words, p_min_rank_first, p_min_rank_second)

        # Update words.
        p_idxs = tf.boolean_mask(tf.range(tf.shape(mask)[0]), mask)
        tensor_words = words.to_tensor(default_value="")
        tensor_p_words = p_words.to_tensor(
            default_value="", 
            shape=[tf.shape(p_idxs)[0], tf.shape(tensor_words)[1]]
        )
        words = tf.tensor_scatter_nd_update(
            tensor_words, 
            tf.expand_dims(p_idxs, axis=1), 
            tensor_p_words,
        )
        words = self._remove_empty_strings(words)
        return [words, mask]
        
    def _byte_pair_encoding(self, tokens):
        """Process unseen tokens and add to cache."""
        words = self._encode_tokens(tokens)
        num_words = tf.shape(words)[0]

        # Merge bytes.
        loop_condition = lambda _, mask : tf.math.count_nonzero(mask) > 0
        initial_mask = tf.fill((num_words,), True)
        merged_words, _ = tf.while_loop(
            loop_condition,
            self._byte_pair_merge_loop_body,
            loop_vars=[words, initial_mask],
            shape_invariants=[
                tf.TensorShape([None, None]),
                tf.TensorShape([None]),
            ]
        )

        merged_words_hash = tf.strings.reduce_join(merged_words, axis=1, separator=" ")
        self.cache.insert(tokens, merged_words_hash)

