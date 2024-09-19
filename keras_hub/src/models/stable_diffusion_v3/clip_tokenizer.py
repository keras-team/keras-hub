# Copyright 2024 The KerasHub Authors
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
from keras_hub.src.tokenizers.byte_pair_tokenizer import BytePairTokenizer
from keras_hub.src.tokenizers.byte_pair_tokenizer import convert_to_ragged_batch
from keras_hub.src.tokenizers.byte_pair_tokenizer import split_strings_for_bpe

try:
    import tensorflow as tf
except ImportError:
    tf = None


class CLIPTokenizer(BytePairTokenizer):
    def __init__(self, vocabulary=None, merges=None, **kwargs):
        self.start_token = "<|startoftext|>"
        self.end_token = "<|endoftext|>"

        super().__init__(
            vocabulary=vocabulary,
            merges=merges,
            unsplittable_tokens=[self.start_token, self.end_token],
            **kwargs,
        )

    def set_vocabulary_and_merges(self, vocabulary, merges):
        super().set_vocabulary_and_merges(vocabulary, merges)

        if vocabulary is not None:
            # Check for necessary special tokens.
            if self.end_token not in self.get_vocabulary():
                raise ValueError(
                    f"Cannot find token `'{self.end_token}'` in the provided "
                    f"`vocabulary`. Please provide `'{self.end_token}'` in "
                    "your `vocabulary` or use a pretrained `vocabulary` name."
                )

            self.start_token_id = self.token_to_id(self.start_token)
            self.end_token_id = self.token_to_id(self.end_token)
            self.pad_token_id = 0
        else:
            self.end_token_id = None
            self.start_token_id = None
            self.pad_token_id = None

    def _bpe_merge_and_update_cache(self, tokens):
        """Process unseen tokens and add to cache."""
        words = self._transform_bytes(tokens)

        # In StableDiffusionV3, we need to add `</w>` to the last word.
        words = tf.strings.reduce_join(words, axis=1, separator=" ")
        words = tf.strings.join([words, "</w>"])
        words = tf.strings.split(words, sep=" ")

        tokenized_words = self._bpe_merge(words)

        # For each word, join all its token by a whitespace,
        # e.g., ["dragon", "fly"] => "dragon fly" for hash purpose.
        tokenized_words = tf.strings.reduce_join(
            tokenized_words, axis=1, separator=" "
        )
        self.cache.insert(tokens, tokenized_words)

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

        # Strip and remove empty tokens.
        raw_tokens = tf.strings.strip(raw_tokens)
        raw_tokens = tf.ragged.boolean_mask(raw_tokens, raw_tokens != "")

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

        # When detokenizing, we need to remove </w> and extra whitespace.
        unicode_text = tf.strings.regex_replace(unicode_text, r"</w>", " ")
        unicode_text = tf.strings.strip(unicode_text)

        split_unicode_text = tf.strings.unicode_split(unicode_text, "UTF-8")
        outputs = tf.strings.reduce_join(
            self.unicode2byte.lookup(split_unicode_text), axis=-1
        )

        if unbatched:
            outputs = tf.squeeze(outputs, 0)
        return outputs

    def get_config(self):
        config = super().get_config()
        # In the constructor, we pass the list of special tokens to the
        # `unsplittable_tokens` arg of the superclass' constructor. Hence, we
        # delete it from the config here.
        del config["unsplittable_tokens"]
        return config
