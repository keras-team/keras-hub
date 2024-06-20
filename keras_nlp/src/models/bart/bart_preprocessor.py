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


import keras

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.layers.preprocessing.start_end_packer import StartEndPacker
from keras_nlp.src.models.bart.bart_tokenizer import BartTokenizer
from keras_nlp.src.models.preprocessor import Preprocessor
from keras_nlp.src.utils.keras_utils import (
    convert_inputs_to_list_of_tensor_segments,
)


@keras_nlp_export("keras_nlp.models.BartPreprocessor")
class BartPreprocessor(Preprocessor):
    """A BART preprocessing layer which tokenizes and packs inputs.

    This preprocessing layer will do three things:

     1. Tokenize both encoder inputs and decoder inputs using the `tokenizer`.
        Both inputs can contain only one segment.
     2. Add the appropriate special tokens - `"<s>"`, `"</s>"` and `"<pad>"`.
     3. Construct a dictionary with keys `"encoder_token_ids"`,
        `"encoder_padding_mask"`, `"decoder_token_ids"`, `"decoder_padding_mask"`
        that can be passed directly to a BART model.

    Args:
        tokenizer: A `keras_nlp.models.BartTokenizer` instance.
        encoder_sequence_length: The length of the packed encoder inputs.
        decoder_sequence_length: The length of the packed decoder inputs.

    Call arguments:
        x: A dictionary with `encoder_text` and `decoder_text` as its keys.
            Each value in the dictionary should be a tensor of single string
            sequences. Inputs may be batched or unbatched. Raw python inputs
            will be converted to tensors.
        y: Any label data. Will be passed through unaltered.
        sample_weight: Any label weight data. Will be passed through unaltered.

    Examples:

    Directly calling the layer on data.
    ```python
    preprocessor = keras_nlp.models.BartPreprocessor.from_preset("bart_base_en")

    # Preprocess unbatched inputs.
    inputs = {
        "encoder_text": "The fox was sleeping.",
        "decoder_text": "The fox was awake."
    }
    preprocessor(inputs)

    # Preprocess batched inputs.
    inputs = {
        "encoder_text": ["The fox was sleeping.", "The lion was quiet."],
        "decoder_text": ["The fox was awake.", "The lion was roaring."]
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
        "encoder_text": "The fox was sleeping.",
        "decoder_text": "The fox was awake."
    }
    preprocessor(inputs)
    ```

    Mapping with `tf.data.Dataset`.
    ```python
    preprocessor = keras_nlp.models.BartPreprocessor.from_preset("bart_base_en")

    # Map labeled single sentences.
    features = {
        "encoder_text": tf.constant(
            ["The fox was sleeping.", "The lion was quiet."]
        ),
        "decoder_text": tf.constant(
            ["The fox was awake.", "The lion was silent."]
        )
    }
    labels = tf.constant(["True", "False"])
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)

    # Map unlabeled single sentences.
    features = {
        "encoder_text": tf.constant(
            ["The fox was sleeping.", "The lion was quiet."]
        ),
        "decoder_text": tf.constant(
            ["The fox was awake.", "The lion was roaring."]
        )
    }
    ds = tf.data.Dataset.from_tensor_slices(features)
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
    ```
    """

    tokenizer_cls = BartTokenizer

    def __init__(
        self,
        tokenizer,
        encoder_sequence_length=1024,
        decoder_sequence_length=1024,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.encoder_packer = None
        self.decoder_packer = None
        self.encoder_sequence_length = encoder_sequence_length
        self.decoder_sequence_length = decoder_sequence_length

    def build(self, input_shape):
        # Defer packer creation to `build()` so that we can be sure tokenizer
        # assets have loaded when restoring a saved model.

        # TODO: Use `MultiSegmentPacker` instead of `StartEndPacker` once we
        # want to move to multi-segment packing and have improved
        # `MultiSegmentPacker`'s performance.
        self.encoder_packer = StartEndPacker(
            start_value=self.tokenizer.start_token_id,
            end_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            sequence_length=self.encoder_sequence_length,
            return_padding_mask=True,
        )

        # The decoder is packed a bit differently; the format is as follows:
        # `[end_token_id, start_token_id, tokens..., end_token_id, padding...]`.
        self.decoder_packer = StartEndPacker(
            start_value=[
                self.tokenizer.end_token_id,
                self.tokenizer.start_token_id,
            ],
            end_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            sequence_length=self.decoder_sequence_length,
            return_padding_mask=True,
        )
        self.built = True

    def call(
        self,
        x,
        y=None,
        sample_weight=None,
        *,
        encoder_sequence_length=None,
        decoder_sequence_length=None,
        # `sequence_length` is an alias for `decoder_sequence_length`
        sequence_length=None,
    ):
        if not (
            isinstance(x, dict)
            and all(k in x for k in ("encoder_text", "decoder_text"))
        ):
            raise ValueError(
                '`x` must be a dictionary, containing the keys `"encoder_text"`'
                f' and `"decoder_text"`. Received x={x}.'
            )

        if encoder_sequence_length is None:
            encoder_sequence_length = self.encoder_sequence_length
        decoder_sequence_length = decoder_sequence_length or sequence_length
        if decoder_sequence_length is None:
            decoder_sequence_length = self.decoder_sequence_length

        encoder_text = x["encoder_text"]
        decoder_text = x["decoder_text"]

        encoder_text = convert_inputs_to_list_of_tensor_segments(encoder_text)
        decoder_text = convert_inputs_to_list_of_tensor_segments(decoder_text)

        if len(encoder_text) > 1 or len(decoder_text) > 1:
            raise ValueError(
                '`BARTPreprocessor` requires both `"encoder_text"` and '
                f'`"decoder_text"` to contain only one segment, but received '
                f"{len(encoder_text)} and {len(decoder_text)}, respectively."
            )

        encoder_inputs = self.tokenizer(encoder_text[0])
        encoder_token_ids, encoder_padding_mask = self.encoder_packer(
            encoder_inputs,
            sequence_length=encoder_sequence_length,
        )

        decoder_inputs = self.tokenizer(decoder_text[0])
        decoder_token_ids, decoder_padding_mask = self.decoder_packer(
            decoder_inputs,
            sequence_length=decoder_sequence_length,
        )

        x = {
            "encoder_token_ids": encoder_token_ids,
            "encoder_padding_mask": encoder_padding_mask,
            "decoder_token_ids": decoder_token_ids,
            "decoder_padding_mask": decoder_padding_mask,
        }

        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "encoder_sequence_length": self.encoder_sequence_length,
                "decoder_sequence_length": self.decoder_sequence_length,
            }
        )
        return config

    @property
    def encoder_sequence_length(self):
        """The padded length of encoder input sequences."""
        return self._encoder_sequence_length

    @encoder_sequence_length.setter
    def encoder_sequence_length(self, value):
        self._encoder_sequence_length = value
        if self.encoder_packer is not None:
            self.encoder_packer.sequence_length = value

    @property
    def decoder_sequence_length(self):
        """The padded length of decoder input sequences."""
        return self._decoder_sequence_length

    @decoder_sequence_length.setter
    def decoder_sequence_length(self, value):
        self._decoder_sequence_length = value
        if self.decoder_packer is not None:
            self.decoder_packer.sequence_length = value

    @property
    def sequence_length(self):
        """Alias for `decoder_sequence_length`."""
        return self.decoder_sequence_length

    @sequence_length.setter
    def sequence_length(self, value):
        self.decoder_sequence_length = value
