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

from typing import Iterable

import tensorflow as tf
import tensorflow_text as tf_text
from absl import logging
from tensorflow import keras


class LMMask(keras.layers.Layer):
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
        vocabulary,
        unselectable_tokens=None,
        mask_token="[MASK]",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.unselectable_tokens = unselectable_tokens
        self.mask_token = mask_token

        # Parse the vocabulary.
        if isinstance(vocabulary, str):
            self._vocab = [
                line.rstrip() for line in tf.io.gfile.GFile(vocabulary)
            ]
        elif isinstance(vocabulary, Iterable):
            self._vocab = list(vocabulary)
        else:
            raise ValueError(
                "Vocabulary must be an file path or list of terms. "
                f"Received: vocabulary={vocabulary}"
            )

        # Parse the ids of unselectable tokens.
        if unselectable_tokens is None:
            self._unselectable_token_ids = []
        else:
            for token in unselectable_tokens:
                if token not in self._vocab:
                    logging.warning(
                        "Unselectable token %s does not exist in the vocab, "
                        "so it is ignored. Please choose unselectable "
                        "tokens from the right vocabulary.",
                        token,
                    )
                else:
                    self._unselectable_token_ids = [
                        self._vocab.index(token)
                        for token in unselectable_tokens
                    ]

        if mask_token not in self._vocab:
            raise KeyError(
                f"Mask token {mask_token} does not exist in the given "
                f"vocabulary, please make sure the mask token is selected "
                f"from the correct vocabulary."
            )
        self._mask_token_id = self._vocab.index(mask_token)

    def call(
        self,
        inputs,
        lm_selection_rate,
        max_selections,
        mask_token_rate=0.8,
        random_token_rate=0.1,
        padding_token=None,
    ):
        if not isinstance(inputs, tf.RaggedTensor):
            # Convert to RaggedTensor to avoid masking out padded token.
            inputs = tf.RaggedTensor.from_tensor(inputs, padding=padding_token)
        random_selector = tf_text.RandomItemSelector(
            max_selections_per_batch=max_selections,
            selection_rate=lm_selection_rate,
            unselectable_ids=self._unselectable_token_ids,
        )
        mask_values_chooser = tf_text.MaskValuesChooser(
            len(self._vocab),
            self._mask_token_id,
            mask_token_rate=mask_token_rate,
            random_token_rate=random_token_rate,
        )
        return tf_text.mask_language_model(
            inputs,
            item_selector=random_selector,
            mask_values_chooser=mask_values_chooser,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary": self._vocab,
                "unselectable_tokens": self.unselectable_tokens,
                "mask_token": self.mask_token,
            }
        )
        return config
