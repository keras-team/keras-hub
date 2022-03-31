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

import tensorflow as tf
import tensorflow_text as tf_text
from tensorflow import keras


class MaskedLanguageModelMasker(keras.layers.Layer):
    """Class that applies language model masking.

    This class is useful for preparing inputs for masked languaged modeling
    (MLM) tasks. It follows the masking strategy described in the [original BERT
    paper](https://arxiv.org/abs/1810.04805). Basically, given a tokenized text,
    it randomly selects certain number of tokens for masking. Then for each
    selected token, it has chance (configurable) to be replaced by "mask token"
    or random token, or stay unchanged.

    This class can both be applied in tf.data pipeline as a standalone utility,
    or used together with `tf.keras.Model` to generate dynamic mask, which is
    useful for workflows like RoBERTa training.

    Args:
        vocabulary: A list of strings or a string string filename path. If
            passing a list, each element of the list should be a single word
            piece token string. If passing a filename, the file should be a
            plain text file containing a single word piece token per line.
        unselectable_tokens: A list of tokens, defaults to None. Tokens in
            `unselectable_tokens` will not be selected for masking.
        mask_token: String, defaults to "[MASK]". The mask token.
    """

    def __init__(
        self,
        vocabulary_size,
        lm_selection_rate,
        max_selections,
        unselectable_token_ids=None,
        mask_token_id=0,
        mask_token_rate=0.8,
        random_token_rate=0.1,
        padding_token=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocabulary_size = vocabulary_size
        self.unselectable_token_ids = unselectable_token_ids
        self.lm_selection_rate = lm_selection_rate
        self.max_selections = max_selections
        self.mask_token_rate = mask_token_rate
        self.random_token_rate = random_token_rate
        self.padding_token = padding_token

        if mask_token_id >= vocabulary_size:
            raise ValueError(
                f"Mask token id should be in range [0, vocabulary_size - 1], "
                f"but received mask_token_id={mask_token_id}."
            )
        self.mask_token_id = mask_token_id

    def call(
        self,
        inputs,
    ):
        input_is_ragged = isinstance(inputs, tf.RaggedTensor)
        if not input_is_ragged:
            # Convert to RaggedTensor to avoid masking out padded token.
            inputs = tf.RaggedTensor.from_tensor(
                inputs,
                padding=self.padding_token,
            )
        random_selector = tf_text.RandomItemSelector(
            max_selections_per_batch=self.max_selections,
            selection_rate=self.lm_selection_rate,
            unselectable_ids=self.unselectable_token_ids,
        )
        mask_values_chooser = tf_text.MaskValuesChooser(
            self.vocabulary_size,
            self.mask_token_id,
            mask_token_rate=self.mask_token_rate,
            random_token_rate=self.random_token_rate,
        )

        (
            masked_input_ids,
            masked_positions,
            masked_ids,
        ) = tf_text.mask_language_model(
            inputs,
            item_selector=random_selector,
            mask_values_chooser=mask_values_chooser,
        )

        if not input_is_ragged:
            # if inputs is a Tensor not RaggedTensor, we format the masked
            # output to be a Tensor.
            masked_input_ids = masked_input_ids.to_tensor()

        return masked_input_ids, masked_positions, masked_ids

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "lm_selection_rate": self.lm_selection_rate,
                "max_selections": self.max_selections,
                "unselectable_token_ids": self.unselectable_token_ids,
                "mask_token_id": self.mask_token_id,
                "mask_token_rate": self.mask_token_rate,
                "random_token_rate": self.random_token_rate,
                "padding_token": self.padding_token,
            }
        )
        return config
