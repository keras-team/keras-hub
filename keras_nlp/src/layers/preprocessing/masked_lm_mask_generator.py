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

try:
    import tensorflow as tf
except ImportError:
    raise ImportError(
        "To use `keras_nlp`, please install Tensorflow: `pip install tensorflow`. "
        "The TensorFlow package is required for data preprocessing with any backend."
    )

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.layers.preprocessing.preprocessing_layer import (
    PreprocessingLayer,
)
from keras_nlp.src.utils.tensor_utils import assert_tf_text_installed
from keras_nlp.src.utils.tensor_utils import convert_to_ragged_batch

try:
    import tensorflow_text as tf_text
except ImportError:
    tf_text = None


@keras_nlp_export("keras_nlp.layers.MaskedLMMaskGenerator")
class MaskedLMMaskGenerator(PreprocessingLayer):
    """Layer that applies language model masking.

    This layer is useful for preparing inputs for masked language modeling
    (MaskedLM) tasks. It follows the masking strategy described in the
    [original BERT paper](https://arxiv.org/abs/1810.04805). Given tokenized
    text, it randomly selects certain number of tokens for masking. Then for
    each selected token, it has a chance (configurable) to be replaced by
    "mask token" or random token, or stay unchanged.

    Input data should be passed as tensors, `tf.RaggedTensor`s, or lists. For
    batched input, inputs should be a list of lists or a rank two tensor. For
    unbatched inputs, each element should be a list or a rank one tensor.

    This layer can be used with `tf.data` to generate dynamic masks on the fly
    during training.

    Args:
        vocabulary_size: int, the size of the vocabulary.
        mask_selection_rate: float, the probability of a token is selected for
            masking.
        mask_token_id: int. The id of mask token.
        mask_selection_length: int. Maximum number of tokens
            selected for  masking in each sequence. If set, the output
            `mask_positions`, `mask_ids` and `mask_weights` will be padded
            to dense tensors of length `mask_selection_length`, otherwise
            the output will be a RaggedTensor. Defaults to `None`.
        unselectable_token_ids: A list of tokens id that should not be
            considered eligible for masking. By default, we assume `0`
            corresponds to a padding token and ignore it. Defaults to `[0]`.
        mask_token_rate: float. `mask_token_rate` must be
            between 0 and 1 which indicates how often the mask_token is
            substituted for tokens selected for masking. Defaults to `0.8`.
        random_token_rate: float. `random_token_rate` must be
            between 0 and 1 which indicates how often a random token is
            substituted for tokens selected for masking.
            Note: mask_token_rate + random_token_rate <= 1,  and for
            (1 - mask_token_rate - random_token_rate), the token will not be
            changed. Defaults to `0.1`.

    Returns:
        A Dict with 4 keys:
            token_ids: Tensor or RaggedTensor, has the same type and shape of
                input. Sequence after getting masked.
            mask_positions: Tensor, or RaggedTensor if `mask_selection_length`
                is None. The positions of token_ids getting masked.
            mask_ids: Tensor, or RaggedTensor if  `mask_selection_length` is
                None. The original token ids at masked positions.
            mask_weights: Tensor, or RaggedTensor if `mask_selection_length` is
                None. `mask_weights` has the same shape as `mask_positions` and
                `mask_ids`. Each element in `mask_weights` should be 0 or 1,
                1 means the corresponding position in `mask_positions` is an
                actual mask, 0 means it is a pad.

    Examples:

    Basic usage.
    ```python
    masker = keras_nlp.layers.MaskedLMMaskGenerator(
        vocabulary_size=10,
        mask_selection_rate=0.2,
        mask_token_id=0,
        mask_selection_length=5
    )
    # Dense input.
    masker([1, 2, 3, 4, 5])

    # Ragged input.
    masker([[1, 2], [1, 2, 3, 4]])
    ```

    Masking a batch that contains special tokens.
    ```python
    pad_id, cls_id, sep_id, mask_id = 0, 1, 2, 3
    batch = [
        [cls_id,   4,    5,      6, sep_id,    7,    8, sep_id, pad_id, pad_id],
        [cls_id,   4,    5, sep_id,      6,    7,    8,      9, sep_id, pad_id],
    ]

    masker = keras_nlp.layers.MaskedLMMaskGenerator(
        vocabulary_size = 10,
        mask_selection_rate = 0.2,
        mask_selection_length = 5,
        mask_token_id = mask_id,
        unselectable_token_ids = [
            cls_id,
            sep_id,
            pad_id,
        ]
    )
    masker(batch)
    ```
    """

    def __init__(
        self,
        vocabulary_size,
        mask_selection_rate,
        mask_token_id,
        mask_selection_length=None,
        unselectable_token_ids=[0],
        mask_token_rate=0.8,
        random_token_rate=0.1,
        **kwargs,
    ):
        assert_tf_text_installed(self.__class__.__name__)

        super().__init__(**kwargs)

        self.vocabulary_size = vocabulary_size
        self.unselectable_token_ids = unselectable_token_ids
        self.mask_selection_rate = mask_selection_rate
        self.mask_selection_length = mask_selection_length
        self.mask_token_rate = mask_token_rate
        self.random_token_rate = random_token_rate

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
        inputs, unbatched, rectangular = convert_to_ragged_batch(inputs)

        (
            token_ids,
            mask_positions,
            mask_ids,
        ) = tf_text.mask_language_model(
            inputs,
            item_selector=self._random_selector,
            mask_values_chooser=self._mask_values_chooser,
        )

        if rectangular:
            # If we converted the input from dense to ragged, convert back.
            token_ids = token_ids.to_tensor()

        mask_weights = tf.ones_like(mask_positions, self.compute_dtype)
        # If `mask_selection_length` is set, convert to dense.
        if self.mask_selection_length:
            target_shape = tf.cast([-1, self.mask_selection_length], "int64")
            mask_positions = mask_positions.to_tensor(shape=target_shape)
            mask_ids = mask_ids.to_tensor(shape=target_shape)
            mask_weights = mask_weights.to_tensor(shape=target_shape)

        if unbatched:
            # If inputs is 1D, we format the output to be 1D as well.
            token_ids = tf.squeeze(token_ids, axis=0)
            mask_positions = tf.squeeze(mask_positions, axis=0)
            mask_ids = tf.squeeze(mask_ids, axis=0)
            mask_weights = tf.squeeze(mask_weights, axis=0)

        return {
            "token_ids": token_ids,
            "mask_positions": mask_positions,
            "mask_ids": mask_ids,
            "mask_weights": mask_weights,
        }

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
            }
        )
        return config
