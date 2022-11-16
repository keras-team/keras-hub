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
from keras_nlp.utils.python_utils import classproperty


@keras.utils.register_keras_serializable(package="keras_nlp")
class DistilBertTokenizer(WordPieceTokenizer):
    """A DistilBERT tokenizer using WordPiece subword segmentation.

    This tokenizer class will tokenize raw strings into integer sequences and
    is based on `keras_nlp.tokenizers.WordPieceTokenizer`. Unlike the
    underlying tokenizer, it will check for all special tokens needed by DistilBERT
    models and provides a `from_preset()` method to automatically download
    a matching vocabulary for a DistilBERT preset.

    This tokenizer does not provide truncation or padding of inputs. It can be
    combined with a `keras_nlp.models.DistilBertPreprocessor` layer for input packing.

    If input is a batch of strings (rank > 0), the layer will output a
    `tf.RaggedTensor` where the last dimension of the output is ragged.

    If input is a scalar string (rank == 0), the layer will output a dense
    `tf.Tensor` with static shape `[None]`.

    Args:
        vocabulary: A list of strings or a string filename path. If
            passing a list, each element of the list should be a single word
            piece token string. If passing a filename, the file should be a
            plain text file containing a single word piece token per line.
        lowercase: If true, the input text will be first lowered before
            tokenization.

    Examples:

    Batched input.
    >>> vocab = ["[UNK]", "[CLS]", "[SEP]", "[PAD]"]
    >>> vocab += ["The", "qu", "##ick", "brown", "fox", "."]
    >>> inputs = ["The quick brown fox.", "The fox."]
    >>> tokenizer = keras_nlp.models.DistilBertTokenizer(vocabulary=vocab)
    >>> tokenizer(inputs)
    <tf.RaggedTensor [[4, 5, 6, 7, 8, 9], [4, 8, 9]]>

    Unbatched input.
    >>> vocab = ["[UNK]", "[CLS]", "[SEP]", "[PAD]"]
    >>> vocab += ["The", "qu", "##ick", "brown", "fox", "."]
    >>> inputs = "The fox."
    >>> tokenizer = keras_nlp.models.DistilBertTokenizer(vocabulary=vocab)
    >>> tokenizer(inputs)
    <tf.Tensor: shape=(3,), dtype=int32, numpy=array([4, 8, 9], dtype=int32)>

    Detokenization.
    >>> vocab = ["[UNK]", "[CLS]", "[SEP]", "[PAD]"]
    >>> vocab += ["The", "qu", "##ick", "brown", "fox", "."]
    >>> inputs = "The quick brown fox."
    >>> tokenizer = keras_nlp.models.DistilBertTokenizer(vocabulary=vocab)
    >>> tokenizer.detokenize(tokenizer.tokenize(inputs)).numpy().decode('utf-8')
    'The quick brown fox .'
    """

    def __init__(
        self,
        vocabulary,
        lowercase=False,
        **kwargs,
    ):
        super().__init__(
            vocabulary=vocabulary,
            lowercase=lowercase,
            **kwargs,
        )

        # Check for necessary special tokens.
        cls_token = "[CLS]"
        sep_token = "[SEP]"
        pad_token = "[PAD]"
        for token in [cls_token, pad_token, sep_token]:
            if token not in self.get_vocabulary():
                raise ValueError(
                    f"Cannot find token `'{token}'` in the provided "
                    f"`vocabulary`. Please provide `'{token}'` in your "
                    "`vocabulary` or use a pretrained `vocabulary` name."
                )

        self.cls_token_id = self.token_to_id(cls_token)
        self.sep_token_id = self.token_to_id(sep_token)
        self.pad_token_id = self.token_to_id(pad_token)

    @classproperty
    def presets(cls):
        raise NotImplementedError

    @classmethod
    def from_preset(
        cls,
        preset,
        **kwargs,
    ):
        raise NotImplementedError


@keras.utils.register_keras_serializable(package="keras_nlp")
class DistilBertPreprocessor(keras.layers.Layer):
    """A DistilBERT preprocessing layer which tokenizes and packs inputs.

    This preprocessing layer will do three things:

     - Tokenize any number of inputs using the `tokenizer`.
     - Pack the inputs together using a `keras_nlp.layers.MultiSegmentPacker`.
       with the appropriate `"[CLS]"`, `"[SEP]"` and `"[PAD]"` tokens.
     - Construct a dictionary of with keys `"token_ids"` and `"padding_mask"`,
       that can be passed directly to a DistilBERT model.

    This layer will accept either a tuple of (possibly batched) inputs, or a
    single input tensor. If a single tensor is passed, it will be packed
    equivalently to a tuple with a single element.

    Args:
        tokenizer: A `keras_nlp.models.DistilBertTokenizer` instance.
        sequence_length: The length of the packed inputs.
        truncate: string. The algorithm to truncate a list of batched segments
            to fit within `sequence_length`. The value can be either
            `round_robin` or `waterfall`:
                - `"round_robin"`: Available space is assigned one token at a
                    time in a round-robin fashion to the inputs that still need
                    some, until the limit is reached.
                - `"waterfall"`: The allocation of the budget is done using a
                    "waterfall" algorithm that allocates quota in a
                    left-to-right manner and fills up the buckets until we run
                    out of budget. It supports an arbitrary number of segments.

    Examples:
    ```python
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
    vocab += ["The", "qu", "##ick", "br", "##own", "fox", "tripped"]
    vocab += ["Call", "me", "Ish", "##mael", "."]
    vocab += ["Oh", "look", "a", "whale"]
    vocab += ["I", "forgot", "my", "home", "##work"]
    tokenizer = keras_nlp.models.DistilBertTokenizer(vocabulary=vocab)
    preprocessor = keras_nlp.models.DistilBertPreprocessor(tokenizer)

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
        (
            (first_sentences, second_sentences), labels
        )
    )
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
        self.packer = MultiSegmentPacker(
            start_value=self.tokenizer.cls_token_id,
            end_value=self.tokenizer.sep_token_id,
            pad_value=self.tokenizer.pad_token_id,
            truncate=truncate,
            sequence_length=sequence_length,
        )

    @property
    def tokenizer(self):
        """The `keras_nlp.models.DistilBertTokenizer` used to tokenize strings."""
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
        token_ids, _ = self.packer(inputs)
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
