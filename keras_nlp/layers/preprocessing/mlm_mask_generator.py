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


class MLMMaskGenerator(keras.layers.Layer):
    """Layer that applies language model masking.

    This layer is useful for preparing inputs for masked languaged modeling
    (MLM) tasks. It follows the masking strategy described in the [original BERT
    paper](https://arxiv.org/abs/1810.04805). Given tokenized text,
    it randomly selects certain number of tokens for masking. Then for each
    selected token, it has chance (configurable) to be replaced by "mask token"
    or random token, or stay unchanged.

    This layer can both be applied in tf.data pipeline as a standalone utility,
    or used together with `tf.keras.Model` to generate dynamic mask, which is
    useful for workflows like RoBERTa training.

    Args:
        vocabulary_size: int, the size of the vocabulary.
        mask_selection_rate: float, the probability of a token is selected for
            masking.
        mask_selection_length: int, defaults to None. Maximum number of tokens
            selected for  masking in each sequence. If set, the output
            `mask_positions`, `mask_ids` and `mask_weights` will be padded
            to dense tensors of length `mask_selection_length`,
            otherwise the output will be a RaggedTensor.
        unselectable_token_ids: A list of tokens, defaults to None. Tokens in
            `unselectable_tokens_ids` will not be selected for masking.
        mask_token_id: int, defaults to 0. The id of mask token.
        mask_token_rate: float, defaults to 0.8. `mask_token_rate` must be
            between 0 and 1 which indicates how often the mask_token is
            substituted for tokens selected for masking.
        random_token_rate: float, defaults to 0.1. `random_token_rate` must be
            between 0 and 1 which indicates how often a random token is
            substituted for tokens selected for masking. Default is 0.1.
            Note: mask_token_rate + random_token_rate <= 1.
        padding_token_id: int, defaults to None. The id of padding token.

    Input:
        A 1D integer tensor of shape [sequence_length,] or a 2D integer tensor
        of shape [batch_size, sequence_length], or a 2D integer RaggedTensor.
        Represents the sequence to mask.

    Returns:
        A Dict with 4 keys:
            tokens: Tensor, has the same type and shape of input.
                Sequence after getting masked.
            mask_positions: Tensor, or RaggedTensor if
                `mask_selection_length` is None. The positions of tokens
                getting masked.
            mask_ids: Tensor, or RaggedTensor if
                `mask_selection_length` is None. The original token ids
                at masked positions.
            mask_weights: Tensor, or RaggedTensor if
                `mask_selection_length` is None. `mask_weights` has the same
                shape as `mask_positions` and `mask_ids`. Each element in
                `mask_weights` should be 0 or 1, 1 means the corresponding
                position in `mask_positions` is an actual mask, 0 means it is
                a pad.

    Examples:

    Basic usage.
    >>> masker = keras_nlp.preprocessing.MLMMaskGenerator( \
    ...     10, mask_selection_rate=0.2, mask_selection_length=5)
    >>> masker(tf.constant([1, 2, 3, 4, 5]))
    {'tokens': <tf.Tensor: shape=(5,), dtype=int32,
        numpy=array([1, 2, 3, 4, 0], dtype=int32)>,
    'mask_positions': <tf.Tensor: shape=(5,), dtype=int64,
        numpy=array([4, 0, 0, 0, 0])>,
    'mask_ids': <tf.Tensor: shape=(5,), dtype=int32,
        numpy=array([5, 0, 0, 0, 0], dtype=int32)>,
    'mask_weights': <tf.Tensor: shape=(1, 5), dtype=int64,
        numpy=array([[1, 0, 0, 0, 0]])>}

    Ragged Input:
    >>> masker = keras_nlp.preprocessing.MLMMaskGenerator( \
    ...     10, mask_selection_rate=0.5, mask_selection_length=5)
    >>> masker(tf.ragged.constant([[1, 2], [1, 2, 3, 4]]))
    {'tokens': <tf.RaggedTensor [[1, 4], [0, 2, 3, 0]]>,
    'mask_positions': <tf.Tensor: shape=(2, 5), dtype=int64, numpy=
        array([[1, 0, 0, 0, 0],
               [0, 3, 0, 0, 0]])>,
    'mask_ids': <tf.Tensor: shape=(2, 5), dtype=int32, numpy=
        array([[2, 0, 0, 0, 0],
               [1, 4, 0, 0, 0]], dtype=int32)>,
    'mask_weights': <tf.Tensor: shape=(2, 5), dtype=int64, numpy=
        array([[1, 0, 0, 0, 0],
               [1, 1, 0, 0, 0]])>}
    """

    def __init__(
        self,
        vocabulary_size,
        mask_selection_rate,
        mask_selection_length=None,
        unselectable_token_ids=None,
        mask_token_id=0,
        mask_token_rate=0.8,
        random_token_rate=0.1,
        padding_token_id=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocabulary_size = vocabulary_size
        self.unselectable_token_ids = unselectable_token_ids
        self.mask_selection_rate = mask_selection_rate
        self.mask_selection_length = mask_selection_length
        self.mask_token_rate = mask_token_rate
        self.random_token_rate = random_token_rate
        self.padding_token_id = padding_token_id

        if mask_token_id >= vocabulary_size:
            raise ValueError(
                f"Mask token id should be in range [0, vocabulary_size - 1], "
                f"but received mask_token_id={mask_token_id}."
            )
        self.mask_token_id = mask_token_id

        max_selections = self.mask_selection_length
        if max_selections is None:
            # Set a large number to remove the `max_selections_per_batch` cap.
            max_selections = 2**31 - 1
        self._random_selector = tf_text.RandomItemSelector(
            max_selections_per_batch=max_selections,
            selection_rate=self.mask_selection_rate,
            unselectable_ids=self.unselectable_token_ids,
        )
        self._mask_values_chooser = tf_text.MaskValuesChooser(
            self.vocabulary_size,
            self.mask_token_id,
            mask_token_rate=self.mask_token_rate,
            random_token_rate=self.random_token_rate,
        )

    def call(self, inputs):
        input_is_ragged = isinstance(inputs, tf.RaggedTensor)
        input_is_1d = tf.rank(inputs) == 1
        if input_is_1d:
            # If inputs is of rank 1, we manually add the batch axis.
            inputs = inputs[tf.newaxis, :]
        if not input_is_ragged:
            # Convert to RaggedTensor to avoid masking out padded token.
            inputs = tf.RaggedTensor.from_tensor(
                inputs,
                padding=self.padding_token_id,
            )
        (tokens, mask_positions, mask_ids,) = tf_text.mask_language_model(
            inputs,
            item_selector=self._random_selector,
            mask_values_chooser=self._mask_values_chooser,
        )

        if not input_is_ragged:
            # If inputs is a Tensor not RaggedTensor, we format the masked
            # output to be a Tensor.
            tokens = tokens.to_tensor()

        mask_weights = tf.ones_like(mask_positions, self.compute_dtype)
        # If mask_selection_length is set, covert to raggeds to dense.
        if self.mask_selection_length:
            target_shape = tf.cast([-1, self.mask_selection_length], tf.int64)
            mask_positions = mask_positions.to_tensor(shape=target_shape)
            mask_ids = mask_ids.to_tensor(shape=target_shape)
            mask_weights = mask_weights.to_tensor(shape=target_shape)

        if input_is_1d:
            # If inputs is 1D, we format the output to be 1D as well.
            tokens = tf.squeeze(tokens)
            if isinstance(mask_positions, tf.RaggedTensor):
                mask_positions = mask_positions.to_tensor()
                mask_ids = mask_ids.to_tensor()
                mask_weights = mask_weights.to_tensor()
            mask_positions = tf.squeeze(mask_positions)
            mask_ids = tf.squeeze(mask_ids)
            mask_weights = tf.squeeze(mask_weights)

        output_dict = {
            "tokens": tokens,
            "mask_positions": mask_positions,
            "mask_ids": mask_ids,
            "mask_weights": mask_weights,
        }
        return output_dict

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocabulary_size": self.vocabulary_size,
                "mask_selection_rate": self.mask_selection_rate,
                "mask_selection_length": self.mask_selection_length,
                "unselectable_token_ids": self.unselectable_token_ids,
                "mask_token_id": self.mask_token_id,
                "mask_token_rate": self.mask_token_rate,
                "random_token_rate": self.random_token_rate,
                "padding_token_id": self.padding_token_id,
            }
        )
        return config
