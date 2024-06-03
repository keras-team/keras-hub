# Copyright 2024 The KerasNLP Authors
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
from absl import logging

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.models.gemma.gemma_preprocessor import GemmaPreprocessor
from keras_nlp.src.utils.keras_utils import (
    convert_inputs_to_list_of_tensor_segments,
)
from keras_nlp.src.utils.tensor_utils import strip_to_ragged


@keras_nlp_export("keras_nlp.models.GemmaCausalLMPreprocessor")
class GemmaCausalLMPreprocessor(GemmaPreprocessor):
    """Gemma Causal LM preprocessor.

    This preprocessing layer is meant for use with
    `keras_nlp.models.GemmaCausalLM`. By default, it will take in batches of
    strings, and return outputs in a `(x, y, sample_weight)` format, where the
    `y` label is the next token id in the `x` sequence.

    For use with generation, the layer also exposes two methods
    `generate_preprocess()` and `generate_postprocess()`. When this preprocessor
    is attached to a `keras_nlp.models.GemmaCausalLM` instance, these methods
    will be called implicitly in `generate()`. They can also be called
    standalone (e.g. to precompute preprocessing inputs for generation in a
    separate process).

    Args:
        tokenizer: A `keras_nlp.models.GemmaTokenizer` instance.
        sequence_length: The length of the packed inputs.
        add_start_token: If `True`, the preprocessor will prepend the tokenizer
            start token to each input sequence.
        add_end_token: If `True`, the preprocessor will append the tokenizer
            end token to each input sequence.

    Call arguments:
        x: A string, `tf.Tensor` or list of python strings.
        y: Label data. Should always be `None` as the layer generates labels.
        sample_weight: Label weights. Should always be `None` as the layer
            generates label weights.
        sequence_length: Pass to override the configured `sequence_length` of
            the layer.

    Examples:
    ```python
    # Load the preprocessor from a preset.
    preprocessor = keras_nlp.models.GemmaCausalLMPreprocessor.from_preset(
        "gemma_2b_en"
    )

    # Tokenize and pack a single sentence.
    preprocessor("The quick brown fox jumped.")

    # Tokenize a batch of sentences.
    preprocessor(["The quick brown fox jumped.", "Call me Ishmael."])

    # Apply tokenization to a `tf.data.Dataset`.
    features = tf.constant(["The quick brown fox.", "Call me Ishmael."])
    ds = tf.data.Dataset.from_tensor_slices(features)
    ds = ds.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)

    # Prepare tokens for generation (no end token).
    preprocessor.generate_preprocess(["The quick brown fox jumped."])

    # Map generation outputs back to strings.
    preprocessor.generate_postprocess({
        'token_ids': np.array([[2, 714, 4320, 8426, 25341, 32292, 235265, 0]]),
        'padding_mask': np.array([[ 1,  1,  1,  1,  1,  1,  1, 0]]),
    })
    ```
    """

    def call(
        self,
        x,
        y=None,
        sample_weight=None,
        sequence_length=None,
    ):
        if y is not None or sample_weight is not None:
            logging.warning(
                "`GemmaCausalLMPreprocessor` generates `y` and `sample_weight` "
                "based on your input data, but your data already contains `y` "
                "or `sample_weight`. Your `y` and `sample_weight` will be "
                "ignored."
            )
        sequence_length = sequence_length or self.sequence_length

        x = convert_inputs_to_list_of_tensor_segments(x)[0]
        x = self.tokenizer(x)
        # Pad with one extra token to account for the truncation below.
        token_ids, padding_mask = self.packer(
            x,
            sequence_length=sequence_length + 1,
            add_start_value=self.add_start_token,
            add_end_value=self.add_end_token,
        )
        # The last token does not have a next token, so we truncate it out.
        x = {
            "token_ids": token_ids[..., :-1],
            "padding_mask": padding_mask[..., :-1],
        }
        # Target `y` will be the next token.
        y, sample_weight = token_ids[..., 1:], padding_mask[..., 1:]
        return keras.utils.pack_x_y_sample_weight(x, y, sample_weight)

    def generate_preprocess(
        self,
        x,
        sequence_length=None,
    ):
        """Convert strings to integer token input for generation.

        Similar to calling the layer for training, this method takes in strings
        or tensor strings, tokenizes and packs the input, and computes a padding
        mask masking all inputs not filled in with a padded value.

        Unlike calling the layer for training, this method does not compute
        labels and will never append a `tokenizer.end_token_id` to the end of
        the sequence (as generation is expected to continue at the end of the
        inputted prompt).
        """
        if not self.built:
            self.build(None)

        x = convert_inputs_to_list_of_tensor_segments(x)[0]
        x = self.tokenizer(x)
        token_ids, padding_mask = self.packer(
            x, sequence_length=sequence_length, add_end_value=False
        )
        return {
            "token_ids": token_ids,
            "padding_mask": padding_mask,
        }

    def generate_postprocess(self, x):
        """Convert integer token output to strings for generation.

        This method reverses `generate_preprocess()`, by first removing all
        padding and start/end tokens, and then converting the integer sequence
        back to a string.
        """
        if not self.built:
            self.build(None)

        token_ids, padding_mask = x["token_ids"], x["padding_mask"]
        ids_to_strip = (
            self.tokenizer.start_token_id,
            self.tokenizer.end_token_id,
            self.tokenizer.pad_token_id,
        )
        token_ids = strip_to_ragged(token_ids, padding_mask, ids_to_strip)
        return self.tokenizer.detokenize(token_ids)
