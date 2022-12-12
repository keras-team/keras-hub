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
"""DistilBERT preprocessor layers."""

import copy

from tensorflow import keras

from keras_nlp.layers.multi_segment_packer import MultiSegmentPacker
from keras_nlp.models.distil_bert.distil_bert_presets import backbone_presets
from keras_nlp.models.distil_bert.distil_bert_tokenizer import (
    DistilBertTokenizer,
)
from keras_nlp.utils.keras_utils import (
    convert_inputs_to_list_of_tensor_segments,
)
from keras_nlp.utils.keras_utils import pack_x_y_sample_weight
from keras_nlp.utils.python_utils import classproperty
from keras_nlp.utils.python_utils import format_docstring


@keras.utils.register_keras_serializable(package="keras_nlp")
class DistilBertPreprocessor(keras.layers.Layer):
    """A DistilBERT preprocessing layer which tokenizes and packs inputs.

    This preprocessing layer will do three things:

     - Tokenize any number of input segments using the `tokenizer`.
     - Pack the inputs together using a `keras_nlp.layers.MultiSegmentPacker`.
       with the appropriate `"[CLS]"`, `"[SEP]"` and `"[PAD]"` tokens.
     - Construct a dictionary of with keys `"token_ids"` and `"padding_mask"`,
       that can be passed directly to a DistilBERT model.

    This layer can be used directly with `tf.data.Dataset.map` to preprocess
    string data in the `(x, y, sample_weight)` format used by
    `keras.Model.fit`.

    The call method of this layer accepts three arguments, `x`, `y`, and
    `sample_weight`. `x` can be a python string or tensor representing a single
    segment, a list of python strings representing a batch of single segments,
    or a list of tensors representing multiple segments to be packed together.
    `y` and `sample_weight` are both optional, can have any format, and will be
    passed through unaltered.

    Special care should be taken when using `tf.data` to map over an unlabeled
    tuple of string segments. `tf.data.Dataset.map` will unpack this tuple
    directly into the call arguments of this layer, rather than forward all
    argument to `x`. To handle this case, it is recommended to  explicitly call
    the layer, e.g. `ds.map(lambda seg1, seg2: preprocessor(x=(seg1, seg2)))`.

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

    # Tokenize and pack a single sentence.
    sentence = tf.constant("The quick brown fox jumped.")
    preprocessor(sentence)
    # Same output.
    preprocessor("The quick brown fox jumped.")

    # Tokenize and a batch of single sentences.
    sentences = tf.constant(
        ["The quick brown fox jumped.", "Call me Ishmael."]
    )
    preprocessor(sentences)
    # Same output.
    preprocessor(
        ["The quick brown fox jumped.", "Call me Ishmael."]
    )

    # Tokenize and pack a sentence pair.
    first_sentence = tf.constant("The quick brown fox jumped.")
    second_sentence = tf.constant("The fox tripped.")
    preprocessor((first_sentence, second_sentence))

    # Map a dataset to preprocess a single sentence.
    features = tf.constant(
        ["The quick brown fox jumped.", "Call me Ishmael."]
    )
    labels = tf.constant([0, 1])
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)

    # Map a dataset to preprocess sentence pairs.
    first_sentences = tf.constant(
        ["The quick brown fox jumped.", "Call me Ishmael."]
    )
    second_sentences = tf.constant(
        ["The fox tripped.", "Oh look, a whale."]
    )
    labels = tf.constant([1, 1])
    ds = tf.data.Dataset.from_tensor_slices(
        (
            (first_sentences, second_sentences), labels
        )
    )
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)

    # Map a dataset to preprocess unlabeled sentence pairs.
    first_sentences = tf.constant(
        ["The quick brown fox jumped.", "Call me Ishmael."]
    )
    second_sentences = tf.constant(
        ["The fox tripped.", "Oh look, a whale."]
    )
    ds = tf.data.Dataset.from_tensor_slices((first_sentences, second_sentences))
    # Watch out for tf.data's default unpacking of tuples here!
    # Best to invoke the `preprocessor` directly in this case.
    ds = ds.map(
        lambda s1, s2: preprocessor(x=(s1, s2)),
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

    def call(self, x, y=None, sample_weight=None):
        x = convert_inputs_to_list_of_tensor_segments(x)
        x = [self.tokenizer(segment) for segment in x]
        token_ids, _ = self.packer(x)
        x = {
            "token_ids": token_ids,
            "padding_mask": token_ids != self.tokenizer.pad_token_id,
        }
        return pack_x_y_sample_weight(x, y, sample_weight)

    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)

    @classmethod
    @format_docstring(names=", ".join(backbone_presets))
    def from_preset(
        cls,
        preset,
        sequence_length=None,
        truncate="round_robin",
        **kwargs,
    ):
        """Instantiate DistilBERT preprocessor from preset architecture.

        Args:
            preset: string. Must be one of {{names}}.
            sequence_length: int, optional. The length of the packed inputs.
                Must be equal to or smaller than the `max_sequence_length` of
                the preset. If left as default, the `max_sequence_length` of
                the preset will be used.
            truncate: string. The algorithm to truncate a list of batched
                segments to fit within `sequence_length`. The value can be
                either `round_robin` or `waterfall`:
                    - `"round_robin"`: Available space is assigned one token at
                        a time in a round-robin fashion to the inputs that still
                        need some, until the limit is reached.
                    - `"waterfall"`: The allocation of the budget is done using
                        a "waterfall" algorithm that allocates quota in a
                        left-to-right manner and fills up the buckets until we
                        run out of budget. It supports an arbitrary number of
                        segments.

        Examples:
        ```python
        # Load preprocessor from preset
        preprocessor = keras_nlp.models.DistilBertPreprocessor.from_preset(
            "distil_bert_base_en_uncased",
        )
        preprocessor("The quick brown fox jumped.")

        # Override sequence_length
        preprocessor = keras_nlp.models.DistilBertPreprocessor.from_preset(
            "distil_bert_base_en_uncased",
            sequence_length=64
        )
        preprocessor("The quick brown fox jumped.")
        ```
        """
        if preset not in cls.presets:
            raise ValueError(
                "`preset` must be one of "
                f"""{", ".join(cls.presets)}. Received: {preset}."""
            )

        tokenizer = DistilBertTokenizer.from_preset(preset)

        # Use model's `max_sequence_length` if `sequence_length` unspecified;
        # otherwise check that `sequence_length` not too long.
        metadata = cls.presets[preset]
        max_sequence_length = metadata["config"]["max_sequence_length"]
        if sequence_length is not None:
            if sequence_length > max_sequence_length:
                raise ValueError(
                    f"`sequence_length` cannot be longer than `{preset}` "
                    f"preset's `max_sequence_length` of {max_sequence_length}. "
                    f"Received: {sequence_length}."
                )
        else:
            sequence_length = max_sequence_length

        return cls(
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            truncate=truncate,
            **kwargs,
        )
