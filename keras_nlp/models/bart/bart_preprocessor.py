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
"""BART preprocessor layer."""

import copy

from absl import logging

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.layers.multi_segment_packer import MultiSegmentPacker
from keras_nlp.models.bart.bart_presets import backbone_presets
from keras_nlp.models.bart.bart_tokenizer import BartTokenizer
from keras_nlp.models.preprocessor import Preprocessor
from keras_nlp.utils.keras_utils import (
    convert_inputs_to_list_of_tensor_segments,
)
from keras_nlp.utils.keras_utils import pack_x_y_sample_weight
from keras_nlp.utils.python_utils import classproperty

PRESET_NAMES = ", ".join(list(backbone_presets))


@keras_nlp_export("keras_nlp.models.BartPreprocessor")
class BartPreprocessor(Preprocessor):
    """A BART preprocessing layer which tokenizes and packs inputs.

    This preprocessing layer will do three things:

     - Tokenize any number of input segments using the `tokenizer`.
     - Add the appropriate special tokens - `"<s>"`, `"</s>"` and `"<pad>"`.
     - Construct a dictionary with keys `"encoder_token_ids"`,
       `"encoder_padding_mask"`, `"decoder_token_ids"`, `"decoder_padding_mask"`
       that can be passed directly to a BART model.

    This layer can be used directly with `tf.data.Dataset.map` to preprocess
    string data in the `(x, y, sample_weight)` format used by
    `keras.Model.fit`.

    The call method of this layer accepts three arguments, `x`, `y`, and
    `sample_weight`. `x` should be python dictionary, having "encoder_inputs"
    and "decoder_inputs" as its keys. Each value in the dictionary can be a
    python string or tensor representing a single segment or a list of python
    strings representing a batch of single segments. Any value passed to `y`
    will be ignored; `y` is inferred internally by shifting `x["decoder_inputs"]`
    to the left by one. `sample_weight` is optional, can have any format, and
    will be passed through unaltered.

    Args:
        tokenizer: A `keras_nlp.models.BartTokenizer` instance.
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
    # Load the preprocessor from a preset.
    preprocessor = keras_nlp.models.BartPreprocessor.from_preset("bart_base_en")

    # Tokenize and pack a single sentence.
    inputs = {
        "encoder_inputs": "The fox was sleeping.",
        "decoder_inputs": "The fox was awake."
    }
    preprocessor(inputs)
    # Same output.
    inputs = {
        "encoder_inputs": tf.constant("The fox was sleeping."),
        "decoder_inputs": tf.constant("The fox was awake.")
    }
    preprocessor(inputs)

    # Tokenize and a batch of single sentences.
    inputs = {
        "encoder_inputs": ["The fox was sleeping.", "The lion was quiet."],
        "decoder_inputs": ["The fox was awake.", "The lion was roaring."]
    }
    preprocessor(inputs)
    # Same output.
    inputs = {
        "encoder_inputs": tf.constant(
            ["The fox was sleeping.", "The lion was quiet."]
        ),
        "decoder_inputs": tf.constant(
            ["The fox was awake.", "The lion was roaring."]
        )
    }
    preprocessor(inputs)

    # Map a dataset to preprocess a single sentence.
    features = {
        "encoder_inputs": tf.constant(
            ["The fox was sleeping.", "The lion was quiet."]
        ),
        "decoder_inputs": tf.constant(
            ["The fox was awake.", "The lion was roaring."]
        )
    }
    ds = tf.data.Dataset.from_tensor_slices(features)
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)

    # Alternatively, you can create a preprocessor from your own vocabulary.
    # The usage is exactly the same as above.
    vocab = {
        "<s>": 0,
        "<pad>": 1,
        "</s>": 2,
        "Ġafter": 5,
        "noon": 6,
        "Ġsun": 7,
    }
    merges = ["Ġ a", "Ġ s", "Ġ n", "e r", "n o", "o n", "Ġs u", "Ġa f", "no on"]
    merges += ["Ġsu n", "Ġaf t", "Ġaft er"]

    tokenizer = keras_nlp.models.BartTokenizer(
        vocabulary=vocab,
        merges=merges,
    )
    preprocessor = keras_nlp.models.BartPreprocessor(
        tokenizer=tokenizer,
        sequence_length=20,
    )
    ```
    """

    def __init__(
        self,
        tokenizer,
        sequence_length=1024,
        truncate="round_robin",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer

        # TODO: Allow users to pass separate `sequence_length`s for encoder and
        # decoder.
        # Note: We use `MultiSegmentPacker` instead of `StartEndPacker` because
        # we might want to support multiple segments in the future (at least for
        # the encoder).
        self.packer = MultiSegmentPacker(
            start_value=self.tokenizer.start_token_id,
            end_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            truncate=truncate,
            sequence_length=sequence_length,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.packer.sequence_length,
                "truncate": self.packer.truncate,
            }
        )
        return config

    def call(self, x, y=None, sample_weight=None):
        if not (
            isinstance(x, dict)
            and ["encoder_inputs", "decoder_inputs"] == list(x.keys())
        ):
            raise ValueError(
                f'`x` must be a dictionary, containing the keys `"encoder_inputs"`'
                f' and `"decoder_inputs"`. Received x={x}.'
            )

        if y is not None:
            logging.warning(
                "You are explicitly passing `y`. However, "
                "`y` is inferred from decoder inputs given in `x`, and will be "
                "ignored."
            )

        encoder_inputs = x["encoder_inputs"]
        decoder_inputs = x["decoder_inputs"]

        encoder_inputs = convert_inputs_to_list_of_tensor_segments(
            encoder_inputs, support_multiple_segments=False
        )
        encoder_inputs = [self.tokenizer(segment) for segment in encoder_inputs]
        encoder_token_ids, _ = self.packer(encoder_inputs)

        decoder_inputs = convert_inputs_to_list_of_tensor_segments(
            decoder_inputs, support_multiple_segments=False
        )
        decoder_inputs = [self.tokenizer(segment) for segment in decoder_inputs]
        decoder_token_ids, _ = self.packer(decoder_inputs)

        x = {
            "encoder_token_ids": encoder_token_ids,
            "encoder_padding_mask": encoder_token_ids
            != self.tokenizer.pad_token_id,
            "decoder_token_ids": decoder_token_ids,
            "decoder_padding_mask": decoder_token_ids
            != self.tokenizer.pad_token_id,
        }

        # Get the labels by shifting the decoder inputs one place to the left.
        if decoder_token_ids.shape.rank == 1:
            y = decoder_token_ids[1:]
        else:
            y = decoder_token_ids[:, 1:]
        return pack_x_y_sample_weight(x, y, sample_weight)

    @classproperty
    def tokenizer_cls(cls):
        return BartTokenizer

    @classproperty
    def presets(cls):
        return copy.deepcopy({**backbone_presets})
