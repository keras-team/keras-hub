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

"""XLM-RoBERTa preprocessing layers."""

import tensorflow as tf
from tensorflow import keras

from keras_nlp.models.roberta.roberta_preprocessing import (
    RobertaMultiSegmentPacker,
)
from keras_nlp.tokenizers.sentence_piece_tokenizer import SentencePieceTokenizer


@keras.utils.register_keras_serializable(package="keras_nlp")
class XLMRobertaPreprocessor(keras.layers.Layer):
    """XLM-RoBERTa preprocessing layer.

    This preprocessing layer will do three things:

    - Tokenize any number of inputs using a
      `keras_nlp.tokenizers.SentencePieceTokenizer`.
    - Pack the inputs together with the appropriate `"<s>"`, `"</s>"` and
      `"<pad>"` tokens, i.e., adding a single `"<s>"` at the start of the
      entire sequence, `"</s></s>"` at the end of each segment, save the last
      and a `"</s>"` at the end of the entire sequence.
    - Construct a dictionary with keys `"token_ids"` and `"padding_mask"`
      that can be passed directly to a XLM-RoBERTa model.

    Note that the original fairseq implementation modifies the indices of the
    SentencePiece tokenizer output. To preserve compatibility, we make the same
    changes, i.e., `"<s>"`, `"<pad>"`, `"</s>"` and `"<unk>"` are mapped to
    0, 1, 2, 3, respectively, and non-special tokens' indices are shifted right
    by one. Keep this in mind if generating your own vocabulary for tokenization.

    This layer will accept either a tuple of (possibly batched) inputs, or a
    single input tensor. If a single tensor is passed, it will be packed
    equivalently to a tuple with a single element.

    The SentencePiece tokenizer can be accessed via the `tokenizer` property on
    this layer, and can be used directly for custom packing on inputs.

    Args:
        proto: Either a `string` path to a SentencePiece proto file, a `bytes`
            object with a serialized SentencePiece proto.
        sequence_length: The length of the packed inputs.
        truncate: The algorithm to truncate a list of batched segments to fit
            within `sequence_length`. The value can be either `round_robin` or
            `waterfall`:
                - `"round_robin"`: Available space is assigned one token at a
                    time in a round-robin fashion to the inputs that still need
                    some, until the limit is reached.
                - `"waterfall"`: The allocation of the budget is done using a
                    "waterfall" algorithm that allocates quota in a
                    left-to-right manner and fills up the buckets until we run
                    out of budget. It supports an arbitrary number of segments.

    Examples:
    ```python
    preprocessor = keras_nlp.models.XLMRobertaPreprocessor(proto="model.spm")

    # Tokenize and pack a single sentence directly.
    preprocessor("The quick brown fox jumped.")

    # Tokenize and pack a multiple sentence directly.
    preprocessor(("The quick brown fox jumped.", "Call me Ishmael."))

    # Map a dataset to preprocess a single sentence.
    features = ["The quick brown fox jumped.", "I forgot my homework."]
    labels = [0, 1]
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    ds = ds.map(
        lambda x, y: (preprocessor(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Map a dataset to preprocess a multiple sentences.
    first_sentences = ["The quick brown fox jumped.", "Call me Ishmael."]
    second_sentences = ["The fox tripped.", "Oh look, a whale."]
    labels = [1, 1]
    ds = tf.data.Dataset.from_tensor_slices(((first_sentences, second_sentences), labels))
    ds = ds.map(
        lambda x, y: (preprocessor(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ```
    """

    def __init__(
        self,
        proto,
        sequence_length=512,
        truncate="round_robin",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.tokenizer = SentencePieceTokenizer(proto=proto)

        # Check for necessary special tokens.
        start_token_id = 0
        pad_token_id = 1
        end_token_id = 2

        self.packer = RobertaMultiSegmentPacker(
            start_value=start_token_id,
            end_value=end_token_id,
            pad_value=pad_token_id,
            truncate=truncate,
            sequence_length=sequence_length,
        )

        self.pad_token_id = pad_token_id

    def vocabulary_size(self):
        """Returns the vocabulary size of the tokenizer."""
        return self.tokenizer.vocabulary_size()

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "proto": self.tokenizer.proto,
                "sequence_length": self.packer.sequence_length,
                "truncate": self.packer.truncate,
            }
        )
        return config

    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        def _tokenize(x):
            tokenized = self.tokenizer(x)

            # In the official SPM proto file, `[unk]`'s ID is 0. Replace that
            # with 2. This will be changed to 3 (`[unk]`'s ID is 3 in the
            # official implementation) after adding by 1.
            tokenized = tf.where(tf.equal(tokenized, 0), 2, tokenized)
            # Shift the tokens IDs by one.
            tokenized = tf.add(tokenized, 1)
            return tokenized

        inputs = [_tokenize(x) for x in inputs]
        token_ids = self.packer(inputs)
        return {
            "token_ids": token_ids,
            "padding_mask": token_ids != self.pad_token_id,
        }

    @classmethod
    def from_preset(
        cls,
        preset,
        sequence_length=None,
        truncate="round_robin",
        **kwargs,
    ):
        raise NotImplementedError
