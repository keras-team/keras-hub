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

"""BART Seq2Seq preprocessor layer."""

from absl import logging

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.models.bart.bart_preprocessor import BartPreprocessor
from keras_nlp.utils.keras_utils import pack_x_y_sample_weight


@keras_nlp_export("keras_nlp.models.BartSeq2SeqPreprocessor")
class BartSeq2SeqPreprocessor(BartPreprocessor):
    """BART Seq2Seq preprocessor.

    This layer is used as preprocessor for seq2seq tasks using the BART model.
    This class subclasses `keras_nlp.models.BartPreprocessor` and keeps most of
    its functionality. It has two changes from the superclass:

     1. Sets the `y` (label) and `sample_weights` fields by shifting the
        decoder input sequence one step towards the left. Both these fields are
        inferred internally, and any passed values will be ignored.
     2. Drops the last token from the decoder input sequence as it does not have
        a successor.

    Args:
        tokenizer: A `keras_nlp.models.BartTokenizer` instance.
        encoder_sequence_length: The length of the packed encoder inputs.
        decoder_sequence_length: The length of the packed decoder inputs.
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

    Call arguments:
        x: A dictionary with `encoder_inputs` and `decoder_inputs` as its keys.
            Each value in the dictionary can be a tensor of single string
            sequences, or a tuple of multiple tensor sequences to be packed
            together. Inputs may be batched or unbatched. For single sequences,
            raw python inputs will be converted to tensors. For multiple
            sequences, pass tensors directly.
        y: Any label data. Any passed value will be ignored since this is
            calculated internally by shifting the decoder input sequence one
            step to the left.
        sample_weight: Any label weight data. Will be ignored since this is
            calculated internally by shifting the padding mask one step to the
            left.

    Examples:

    Directly calling the layer on data
    ```python
    preprocessor = keras_nlp.models.BartPreprocessor.from_preset("bart_base_en")

    # Tokenize and pack a single sentence.
    inputs = {
        "encoder_inputs": "The fox was sleeping.",
        "decoder_inputs": "The fox was awake."
    }
    preprocessor(inputs)

    # Tokenize a batch of single sentences.
    inputs = {
        "encoder_inputs": ["The fox was sleeping.", "The lion was quiet."],
        "decoder_inputs": ["The fox was awake.", "The lion was roaring."]
    }
    preprocessor(inputs)

    # Tokenize and pack a sentence pair.
    inputs = {
        "encoder_inputs": (
            tf.constant("The fox was sleeping."),
            tf.constant("The lion was quiet.")
        ),
        "decoder_inputs": (
            tf.constant("The fox was awake."),
            tf.constant("The lion was roaring.")
        )
    }
    preprocessor(inputs)

    # Custom vocabulary.
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
        encoder_sequence_length=20,
        decoder_sequence_length=10,
    )
    inputs = {
        "encoder_inputs": "The fox was sleeping.",
        "decoder_inputs": "The fox was awake."
    }
    preprocessor(inputs)
    ```

    Mapping with `tf.data.Dataset`.
    ```python
    preprocessor = keras_nlp.models.BartPreprocessor.from_preset("bart_base_en")

    # Map single sentences.
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

    # Map sentence pairs.
    features = {
        "encoder_inputs": (
            tf.constant(
                ["The fox was sleeping.", "The lion was quiet."]
            ),
            tf.constant(
                ["It wanted to get up.", "It wanted to roar."]
            ),
        ),
        "decoder_inputs": (
            tf.constant(
                ["The fox was awake.", "The lion was roaring."]
            ),
            tf.constant(
                ["It wanted to sleep.", "It wanted to shout."]
            ),
        ),
    }
    labels = tf.constant(["True", "False"])
    ds = tf.data.Dataset.from_tensor_slices(features)
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
    ```
    """

    def call(self, x, y=None, sample_weight=None):
        if y is not None or sample_weight is not None:
            logging.warning(
                "`BartSeq2SeqPreprocessor` infers `y` and `sample_weight` "
                "from the provided input data, i.e., `x`. However, non-`None`"
                "values have been passed for `y` or `sample_weight` or both. "
                "These values will be ignored."
            )

        # Since we truncate the last token from `decoder_token_ids`, we need to
        # forcefully set the `decoder_sequence_length` to one greater than the
        # value passed.
        self.decoder_sequence_length += 1

        x = super().call(x)
        decoder_token_ids = x.pop("decoder_token_ids")
        decoder_padding_mask = x.pop("decoder_padding_mask")

        # The last token does not have a next token. Hence, we truncate it.
        x = {
            **x,
            "decoder_token_ids": decoder_token_ids[..., :-1],
            "decoder_padding_mask": decoder_padding_mask[..., :-1],
        }
        # Target `y` will be the decoder input sequence shifted one step to the
        # left (i.e., the next token).
        y = decoder_token_ids[..., 1:]
        sample_weight = decoder_padding_mask[..., 1:]
        return pack_x_y_sample_weight(x, y, sample_weight)
