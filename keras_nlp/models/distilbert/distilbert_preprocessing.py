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
"""DistilBERT preprocessing layers."""

from tensorflow import keras

from keras_nlp.layers.multi_segment_packer import MultiSegmentPacker
from keras_nlp.tokenizers.word_piece_tokenizer import WordPieceTokenizer


@keras.utils.register_keras_serializable(package="keras_nlp")
class DistilBertPreprocessor(keras.layers.Layer):
    """DistilBERT preprocessing layer.

    This preprocessing layer will do three things:

     - Tokenize any number of inputs using a
       `keras_nlp.tokenizers.WordPieceTokenizer`.
     - Pack the inputs together using a `keras_nlp.layers.MultiSegmentPacker`.
       with the appropriate `"[CLS]"`, `"[SEP]"` and `"[PAD]"` tokens.
     - Construct a dictionary of with keys `"token_ids"`, `"padding_mask"`,
       that can be passed directly to a DistilBERT model.

    This layer will accept either a tuple of (possibly batched) inputs, or a
    single input tensor. If a single tensor is passed, it will be packed
    equivalently to a tuple with a single element.

    The WordPiece tokenizer can be accessed via the `tokenizer` property on this
    layer, and can be used directly for custom packing on inputs.

    Args:
        vocabulary: A list of vocabulary terms or a vocabulary filename.
        lowercase: If `True`, input will be lowercase before tokenization. If
            `vocabulary` is set to a pretrained vocabulary, this parameter will
            be inferred.
        sequence_length: The length of the packed inputs. Only used if
            `pack_inputs` is True.
        truncate: The algorithm to truncate a list of batched segments to fit
            within `sequence_length`. Only used if
            `pack_inputs` is True. The value can be either `round_robin` or
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
    preprocessor = keras_nlp.models.DistilBertPreprocessor(
        vocabulary="./vocab.txt",
    )

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
    ds = tf.data.Dataset.from_tensor_slices(
        ((first_sentences, second_sentences), labels))
    ds = ds.map(
        lambda x, y: (preprocessor(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ```
    """

    def __init__(
        self,
        vocabulary="uncased_en",
        lowercase=False,
        sequence_length=512,
        truncate="round_robin",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.tokenizer = WordPieceTokenizer(
            vocabulary=vocabulary,
            lowercase=lowercase,
        )

        # Check for necessary special tokens.
        cls_token = "[CLS]"
        sep_token = "[SEP]"
        pad_token = "[PAD]"
        for token in [cls_token, pad_token, sep_token]:
            if token not in self.tokenizer.get_vocabulary():
                raise ValueError(
                    f"Cannot find token `'{token}'` in the provided "
                    f"`vocabulary`. Please provide `'{token}'` in your "
                    "`vocabulary` or use a pretrained `vocabulary` name."
                )

        self.pad_token_id = self.tokenizer.token_to_id(pad_token)
        self.packer = MultiSegmentPacker(
            start_value=self.tokenizer.token_to_id(cls_token),
            end_value=self.tokenizer.token_to_id(sep_token),
            pad_value=self.pad_token_id,
            truncate=truncate,
            sequence_length=sequence_length,
        )

    def vocabulary_size(self):
        """Returns the vocabulary size of the tokenizer."""
        return self.tokenizer.vocabulary_size()

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary": self.tokenizer.vocabulary,
                "lowercase": self.tokenizer.lowercase,
                "sequence_length": self.packer.sequence_length,
                "truncate": self.packer.truncate,
            }
        )
        return config

    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        inputs = [self.tokenizer(x) for x in inputs]
        token_ids, _ = self.packer(inputs)
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
