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
import keras

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.start_end_packer import StartEndPacker
from keras_hub.src.models.preprocessor import Preprocessor
from keras_hub.src.utils.tensor_utils import preprocessing_function
from keras_hub.src.utils.tensor_utils import strip_to_ragged

try:
    import tensorflow as tf
except ImportError:
    tf = None


@keras_hub_export("keras_hub.models.Seq2SeqLMPreprocessor")
class Seq2SeqLMPreprocessor(Preprocessor):
    """Base class for seq2seq language modeling preprocessing layers.

    `Seq2SeqLMPreprocessor` tasks wrap a `keras_hub.tokenizer.Tokenizer` to
    create a preprocessing layer for seq2seq language modeling tasks. It is
    intended to be paired with a `keras.models.Seq2SeqLM` task.

    All `Seq2SeqLMPreprocessor` layers take inputs a dictionary input with keys
    `"encoder_text"` and `"decoder_text"`.

    This layer will always output a `(x, y, sample_weight)` tuple, where `x`
    is a dictionary with the tokenized inputs, `y` contains the tokens from `x`
    offset by 1, and `sample_weight` marks where `y` contains padded
    values. The exact contents of `x` will vary depending on the model being
    used.

    a `Seq2SeqLMPreprocessor` contains two extra methods, `generate_preprocess`
    and `generate_postprocess` for use with generation. See examples below.

    All `Seq2SeqLMPreprocessor` tasks include a `from_preset()` constructor
    which can be used to load a pre-trained config and vocabularies. You can
    call the `from_preset()` constructor directly on this base class, in which
    case the correct class for you model will be automatically instantiated.

    Examples.
    ```python
    preprocessor = keras_hub.models.Seq2SeqLMPreprocessor.from_preset(
        "bart_base_en",
        encoder_sequence_length=256,
        decoder_sequence_length=256,
    )

    # Tokenize, mask and pack a single sentence.
    x = {
        "encoder_text": "The fox was sleeping.",
        "decoder_text": "The fox was awake.",
    }
    x, y, sample_weight = preprocessor(x)

    # Tokenize and pad/truncate a batch of labeled sentences.
    x = {
        "encoder_text": ["The fox was sleeping."],
        "decoder_text": ["The fox was awake."],
    x, y, sample_weight = preprocessor(x)

    # With a `tf.data.Dataset`.
    ds = tf.data.Dataset.from_tensor_slices(x)
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)

    # Generate preprocess and postprocess.
    x = preprocessor.generate_preprocess(x)  # Tokenized numeric inputs.
    x = preprocessor.generate_postprocess(x)  # Detokenized string outputs.
    ```
    """

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
        self.encoder_packer = StartEndPacker(
            start_value=self.tokenizer.start_token_id,
            end_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            sequence_length=self.encoder_sequence_length,
            return_padding_mask=True,
        )
        self.decoder_packer = StartEndPacker(
            start_value=self.tokenizer.start_token_id,
            end_value=self.tokenizer.end_token_id,
            pad_value=self.tokenizer.pad_token_id,
            sequence_length=self.decoder_sequence_length,
            return_padding_mask=True,
        )
        self.built = True

    @preprocessing_function
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
        if encoder_sequence_length is None:
            encoder_sequence_length = self.encoder_sequence_length
        decoder_sequence_length = decoder_sequence_length or sequence_length
        if decoder_sequence_length is None:
            decoder_sequence_length = self.decoder_sequence_length

        encoder_inputs = self.tokenizer(x["encoder_text"])
        encoder_token_ids, encoder_padding_mask = self.encoder_packer(
            encoder_inputs,
            sequence_length=encoder_sequence_length,
        )
        decoder_inputs = self.tokenizer(x["decoder_text"])
        decoder_token_ids, decoder_padding_mask = self.decoder_packer(
            decoder_inputs,
            sequence_length=decoder_sequence_length + 1,
        )
        x = {
            "encoder_token_ids": encoder_token_ids,
            "encoder_padding_mask": encoder_padding_mask,
            "decoder_token_ids": decoder_token_ids[..., :-1],
            "decoder_padding_mask": decoder_padding_mask[..., :-1],
        }
        # Target `y` will be the decoder input sequence shifted one step to the
        # left (i.e., the next token).
        y = decoder_token_ids[..., 1:]
        sample_weight = decoder_padding_mask[..., 1:]
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

    @preprocessing_function
    def generate_preprocess(
        self,
        x,
        *,
        encoder_sequence_length=None,
        decoder_sequence_length=None,
        # `sequence_length` is an alias for `decoder_sequence_length`
        sequence_length=None,
    ):
        """Convert encoder and decoder input strings to integer token inputs for generation.

        Similar to calling the layer for training, this method takes in a dict
        containing `"encoder_text"` and `"decoder_text"`, with strings or tensor
        strings for values, tokenizes and packs the input, and computes a
        padding mask masking all inputs not filled in with a padded value.

        Unlike calling the layer for training, this method does not compute
        labels and will never append a tokenizer.end_token_id to the end of
        the decoder sequence (as generation is expected to continue at the end
        of the inputted decoder prompt).
        """
        if not self.built:
            self.build(None)

        if isinstance(x, dict):
            encoder_text = x["encoder_text"]
            decoder_text = x["decoder_text"]
        else:
            encoder_text = x
            # Initialize empty prompt for the decoder.
            decoder_text = tf.fill((tf.shape(encoder_text)[0],), "")

        if encoder_sequence_length is None:
            encoder_sequence_length = self.encoder_sequence_length
        decoder_sequence_length = decoder_sequence_length or sequence_length
        if decoder_sequence_length is None:
            decoder_sequence_length = self.decoder_sequence_length

        # Tokenize and pack the encoder inputs.
        encoder_token_ids = self.tokenizer(encoder_text)
        encoder_token_ids, encoder_padding_mask = self.encoder_packer(
            encoder_token_ids,
            sequence_length=encoder_sequence_length,
        )

        # Tokenize and pack the decoder inputs.
        decoder_token_ids = self.tokenizer(decoder_text)
        decoder_token_ids, decoder_padding_mask = self.decoder_packer(
            decoder_token_ids,
            sequence_length=decoder_sequence_length,
            add_end_value=False,
        )

        return {
            "encoder_token_ids": encoder_token_ids,
            "encoder_padding_mask": encoder_padding_mask,
            "decoder_token_ids": decoder_token_ids,
            "decoder_padding_mask": decoder_padding_mask,
        }

    @preprocessing_function
    def generate_postprocess(
        self,
        x,
    ):
        """Convert integer token output to strings for generation.

        This method reverses `generate_preprocess()`, by first removing all
        padding and start/end tokens, and then converting the integer sequence
        back to a string.
        """
        if not self.built:
            self.build(None)

        token_ids, padding_mask = (
            x["decoder_token_ids"],
            x["decoder_padding_mask"],
        )
        ids_to_strip = self.tokenizer.special_token_ids
        token_ids = strip_to_ragged(token_ids, padding_mask, ids_to_strip)
        return self.tokenizer.detokenize(token_ids)

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
