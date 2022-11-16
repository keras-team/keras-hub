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
from keras_nlp.utils.python_utils import classproperty
from keras_nlp.utils.tf_utils import tensor_to_string_list


@keras.utils.register_keras_serializable(package="keras_nlp")
class XLMRobertaTokenizer(SentencePieceTokenizer):
    """XLM-RoBERTa tokenizer layer based on SentencePiece.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_nlp.tokenizers.SentencePieceTokenizer`. Unlike the
    underlying tokenizer, it will check for all special tokens needed by
    XLM-RoBERTa models and provides a `from_preset()` method to automatically
    download a matching vocabulary for a XLM-RoBERTa preset.

    The original fairseq implementation of XLM-RoBERTa modifies the indices of
    the SentencePiece tokenizer output. To preserve compatibility, we make the
    same changes, i.e., `"<s>"`, `"<pad>"`, `"</s>"` and `"<unk>"` are mapped to
    0, 1, 2, 3, respectively, and non-special token indices are shifted right
    by one.

    If input is a batch of strings (rank > 0), the layer will output a
    `tf.RaggedTensor` where the last dimension of the output is ragged.

    If input is a scalar string (rank == 0), the layer will output a dense
    `tf.Tensor` with static shape `[None]`.

    Args:
        proto: Either a `string` path to a SentencePiece proto file, or a
            `bytes` object with a serialized SentencePiece proto. See the
            [SentencePiece repository](https://github.com/google/sentencepiece)
            for more details on the format.

    Examples:

    ```python
    def train_sentencepiece(ds, vocab_size):
        bytes_io = io.BytesIO()
        sentencepiece.SentencePieceTrainer.train(
            sentence_iterator=ds.as_numpy_iterator(),
            model_writer=bytes_io,
            vocab_size=vocab_size,
            model_type="WORD",
            unk_id=0,
            bos_id=1,
            eos_id=2,
        )
        return bytes_io.getvalue()

    ds = tf.data.Dataset.from_tensor_slices(
        ["the quick brown fox", "the earth is round"]
    )

    proto = train_sentencepiece(ds, vocab_size=10)
    tokenizer = keras_nlp.models.XLMRobertaTokenizer(proto=proto)

    # Batched inputs.
    tokenizer(["the quick brown fox", "the earth is round"])

    # Unbatched inputs.
    tokenizer("the quick brown fox")

    # Detokenization.
    tokenizer.detokenize(tf.constant([[0, 4, 9, 5, 7, 2]]))
    ```
    """

    def __init__(self, proto, **kwargs):
        super().__init__(proto=proto, **kwargs)

        # List of special tokens.
        self._vocabulary_prefix = ["<s>", "<pad>", "</s>", "<unk>"]

        # IDs of special tokens.
        self.start_token_id = 0  # <s>
        self.pad_token_id = 1  # <pad>
        self.end_token_id = 2  # </s>
        self.unk_token_id = 3  # <unk>

    def vocabulary_size(self):
        """Get the size of the tokenizer vocabulary."""
        return super().vocabulary_size() + 1

    def get_vocabulary(self):
        """Get the size of the tokenizer vocabulary."""
        vocabulary = tensor_to_string_list(
            self._sentence_piece.id_to_string(
                tf.range(super().vocabulary_size())
            )
        )
        return self._vocabulary_prefix + vocabulary[3:]

    def id_to_token(self, id):
        """Convert an integer id to a string token."""
        if id < len(self._vocabulary_prefix):
            return self._vocabulary_prefix[id]

        return tensor_to_string_list(self._sentence_piece.id_to_string(id - 1))

    def token_to_id(self, token):
        """Convert a string token to an integer id."""
        if token in self._vocabulary_prefix:
            return self._vocabulary_prefix.index(token)

        return int(self._sentence_piece.string_to_id(token).numpy()) + 1

    def tokenize(self, inputs):
        tokens = super().tokenize(inputs)

        # Correct `unk_token_id` (0 -> 3). Note that we do not correct
        # `start_token_id` and `end_token_id`; they are dealt with in
        # `XLMRobertaPreprocessor`.
        tokens = tf.where(tf.equal(tokens, 0), self.unk_token_id - 1, tokens)

        # Shift the tokens IDs right by one.
        return tf.add(tokens, 1)

    def detokenize(self, inputs):
        if inputs.dtype == tf.string:
            return super().detokenize(inputs)

        # Shift the tokens IDs left by one.
        tokens = tf.subtract(inputs, 1)

        # Correct `unk_token_id`, `end_token_id`, `start_token_id`, respectively.
        # Note: The `pad_token_id` is taken as 0 (`unk_token_id`) since the
        # proto does not contain `pad_token_id`. This mapping of the pad token
        # is done automatically by the above subtraction.
        tokens = tf.where(tf.equal(tokens, self.unk_token_id - 1), 0, tokens)
        tokens = tf.where(tf.equal(tokens, self.end_token_id - 1), 2, tokens)
        tokens = tf.where(tf.equal(tokens, self.start_token_id - 1), 1, tokens)

        # Note: Even though we map `"<s>" and `"</s>"` to the correct IDs,
        # the `detokenize` method will return empty strings for these tokens.
        # This is a vagary of the `sentencepiece` library.
        return super().detokenize(tokens)


@keras.utils.register_keras_serializable(package="keras_nlp")
class XLMRobertaPreprocessor(keras.layers.Layer):
    """XLM-RoBERTa preprocessing layer.

    This preprocessing layer will do three things:

    - Tokenize any number of inputs using `tokenizer`.
    - Pack the inputs together with the appropriate `"<s>"`, `"</s>"` and
      `"<pad>"` tokens, i.e., adding a single `"<s>"` at the start of the
      entire sequence, `"</s></s>"` at the end of each segment, save the last
      and a `"</s>"` at the end of the entire sequence.
    - Construct a dictionary with keys `"token_ids"` and `"padding_mask"`
      that can be passed directly to a XLM-RoBERTa model.

    Note that the original fairseq implementation modifies the indices of the
    SentencePiece tokenizer output. To preserve compatibility, we make the same
    changes, i.e., `"<s>"`, `"<pad>"`, `"</s>"` and `"<unk>"` are mapped to
    0, 1, 2, 3, respectively, and non-special token indices are shifted right
    by one. Keep this in mind if generating your own vocabulary for tokenization.

    This layer will accept either a tuple of (possibly batched) inputs, or a
    single input tensor. If a single tensor is passed, it will be packed
    equivalently to a tuple with a single element.

    Args:
        tokenizer: A `keras_nlp.tokenizers.XLMRobertaTokenizer` instance.
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
    tokenizer = keras_nlp.models.XLMRobertaTokenizer(proto="model.spm")
    preprocessor = keras_nlp.models.XLMRobertaPreprocessor(
        tokenizer=tokenizer,
        sequence_length=10,
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
    ds = tf.data.Dataset.from_tensor_slices(((first_sentences, second_sentences), labels))
    ds = ds.map(
        lambda x, y: (preprocessor(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ```
    """

    def __init__(
        self,
        tokenizer,
        sequence_length=512,
        truncate="round_robin",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._tokenizer = tokenizer

        self.packer = RobertaMultiSegmentPacker(
            start_value=self.tokenizer.start_token_id,
            end_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            truncate=truncate,
            sequence_length=sequence_length,
        )

    @property
    def tokenizer(self):
        """The `keras_nlp.models.XLMRobertaTokenizer` used to tokenize strings."""
        return self._tokenizer

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "tokenizer": keras.layers.serialize(self.tokenizer),
                "sequence_length": self.packer.sequence_length,
                "truncate": self.packer.truncate,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        if "tokenizer" in config:
            config["tokenizer"] = keras.layers.deserialize(config["tokenizer"])
        return cls(**config)

    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        inputs = [self.tokenizer(x) for x in inputs]
        token_ids = self.packer(inputs)
        return {
            "token_ids": token_ids,
            "padding_mask": token_ids != self.tokenizer.pad_token_id,
        }

    @classproperty
    def presets(cls):
        raise NotImplementedError

    @classmethod
    def from_preset(
        cls,
        preset,
        sequence_length=None,
        truncate="round_robin",
        **kwargs,
    ):
        raise NotImplementedError
