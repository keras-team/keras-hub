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

"""XLM-RoBERTa preprocessor layer."""

import copy

from tensorflow import keras

from keras_nlp.models.roberta.roberta_preprocessor import (
    RobertaMultiSegmentPacker,
)
from keras_nlp.models.xlm_roberta.xlm_roberta_presets import backbone_presets
from keras_nlp.models.xlm_roberta.xlm_roberta_tokenizer import (
    XLMRobertaTokenizer,
)
from keras_nlp.utils.python_utils import classproperty
from keras_nlp.utils.python_utils import format_docstring


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
        """Instantiate XLM-RoBERTa preprocessor from preset architecture.

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
        preprocessor = keras_nlp.models.XLMRobertaPreprocessor.from_preset(
            "xlm_roberta_base",
        )
        preprocessor("The quick brown fox jumped.")

        # Override sequence_length
        preprocessor = keras_nlp.models.XLMRobertaPreprocessor.from_preset(
            "xlm_roberta_base",
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

        tokenizer = XLMRobertaTokenizer.from_preset(preset)

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
