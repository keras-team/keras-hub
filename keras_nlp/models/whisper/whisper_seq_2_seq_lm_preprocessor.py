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

"""Whisper Seq2Seq LM preprocessor layer."""

import copy

import tensorflow as tf
from absl import logging

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.backend import ops
from keras_nlp.models.whisper.whisper_preprocessor import WhisperPreprocessor
from keras_nlp.models.whisper.whisper_presets import backbone_presets
from keras_nlp.utils.keras_utils import (
    convert_inputs_to_list_of_tensor_segments,
)
from keras_nlp.utils.keras_utils import pack_x_y_sample_weight
from keras_nlp.utils.python_utils import classproperty


@keras_nlp_export("keras_nlp.models.WhisperSeq2SeqLMPreprocessor")
class WhisperSeq2SeqLMPreprocessor(WhisperPreprocessor):
    """Whisper Seq2Seq LM preprocessor.

    This layer is used as preprocessor for seq2seq tasks using the Whisper model.
    This class subclasses `keras_nlp.models.WhisperPreprocessor` and keeps most of
    its functionality. It has two changes from the superclass:

     1. Sets the `y` (label) and `sample_weights` fields by shifting the
        decoder input sequence one step towards the left. Both these fields are
        inferred internally, and any passed values will be ignored.
     2. Drops the last token from the decoder input sequence as it does not have
        a successor.

    Args:
        tokenizer: A `keras_nlp.models.WhisperTokenizer` instance.
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
        x: A dictionary with `encoder_text` and `decoder_text` as its keys.
            Each value in the dictionary should be a tensor of single string
            sequences. Inputs may be batched or unbatched. Raw python inputs
            will be converted to tensors.
        y: Label data. Should always be `None` as the layer generates labels by
            shifting the decoder input sequence one step to the left.
        sample_weight: Label weights. Should always be `None` as the layer
            generates label weights by shifting the padding mask one step to the
            left.
    """

    def __init__(
        self,
        audio_feature_extractor,
        tokenizer,
        decoder_sequence_length=448,
        language=None,
        task=None,
        no_timestamps=True,
        **kwargs,
    ):
        # Since we truncate the last token from `decoder_token_ids`, we need to
        # forcefully set the `decoder_sequence_length` to one greater than the
        # value passed.
        super().__init__(
            audio_feature_extractor=audio_feature_extractor,
            tokenizer=tokenizer,
            decoder_sequence_length=decoder_sequence_length + 1,
            language=language,
            task=task,
            no_timestamps=no_timestamps,
            **kwargs,
        )

        # Maintain a private copy of the sequence lengths for config purposes.
        self._decoder_sequence_length = decoder_sequence_length

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "decoder_sequence_length": self._decoder_sequence_length,
            }
        )
        return config

    def call(self, x, y=None, sample_weight=None):
        if y is not None or sample_weight is not None:
            logging.warning(
                "`WhisperSeq2SeqLMPreprocessor` infers `y` and `sample_weight` "
                "from the provided input data, i.e., `x`. However, non-`None`"
                "values have been passed for `y` or `sample_weight` or both. "
                "These values will be ignored."
            )

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

    @classproperty
    def presets(cls):
        return copy.deepcopy(backbone_presets)

    def generate_preprocess(
        self,
        x,
        sequence_length=None,
    ):
        """Convert encoder and decoder input strings to integer token inputs for generation.

        Similar to calling the layer for training, this method takes in a dict
        containing `"encoder_text"` and `"decoder_text"`, with strings or tensor
        strings for values, tokenizes and packs the input, and computes a
        padding mask masking all inputs not filled in with a padded value.

        Unlike calling the the layer for training, this method does not compute
        labels and will never append a tokenizer.end_token_id to the end of
        the decoder sequence (as generation is expected to continue at the end
        of the inputted decoder prompt).
        """
        # If `sequence_length` is not provided, we use the default value.
        if sequence_length is None:
            sequence_length = self._decoder_sequence_length

        if isinstance(x, dict):
            encoder_audio = x["encoder_audio"]
            decoder_text = x["decoder_text"]
        else:
            encoder_audio = x
            # Initialize empty prompt for the decoder.
            decoder_text = tf.fill((tf.shape(encoder_audio)[0],), "")

        # Compute the log-mel spectrogram of the audio inputs.
        encoder_audio = convert_inputs_to_list_of_tensor_segments(
            encoder_audio
        )[0]
        encoder_features = self.audio_feature_extractor(encoder_audio)

        # Tokenize and pack the decoder inputs.
        decoder_text = convert_inputs_to_list_of_tensor_segments(decoder_text)[
            0
        ]
        decoder_token_ids = self.tokenizer(decoder_text)
        decoder_token_ids, decoder_padding_mask = self.decoder_packer(
            decoder_token_ids,
            sequence_length=sequence_length,
            add_end_value=False,
        )

        return {
            "encoder_features": encoder_features,
            "decoder_token_ids": decoder_token_ids,
            "decoder_padding_mask": decoder_padding_mask,
        }

    def generate_postprocess(
        self,
        x,
    ):
        """Convert integer token output to strings for generation.

        This method reverses `generate_preprocess()`, by first removing all
        padding and start/end tokens, and then converting the integer sequence
        back to a string.
        """
        decoder_token_ids, decoder_padding_mask = (
            x["decoder_token_ids"],
            x["decoder_padding_mask"],
        )
        if not isinstance(decoder_token_ids, tf.Tensor):
            decoder_token_ids = ops.convert_to_numpy(decoder_token_ids)
        if not isinstance(decoder_padding_mask, tf.Tensor):
            decoder_padding_mask = ops.convert_to_numpy(decoder_padding_mask)
        # Strip any special tokens during detokenization, i.e., the start and
        # end markers. In the future, we could make this configurable.
        decoder_padding_mask = (
            decoder_padding_mask
            & (decoder_token_ids != self.tokenizer.eos_token_id)
            & (decoder_token_ids != self.tokenizer.bos_token_id)
        )
        decoder_token_ids = tf.ragged.boolean_mask(
            decoder_token_ids, decoder_padding_mask
        )
        return self.tokenizer.detokenize(decoder_token_ids)
