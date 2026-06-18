import random

import keras
import numpy as np

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.layers.preprocessing.preprocessing_layer import (
    PreprocessingLayer,
)
from keras_hub.src.utils.tensor_utils import (
    convert_preprocessing_outputs_python,
)
from keras_hub.src.utils.tensor_utils import convert_to_list
from keras_hub.src.utils.tensor_utils import convert_to_ragged_batch
from keras_hub.src.utils.tensor_utils import in_tf_function
from keras_hub.src.utils.tensor_utils import preprocessing_function

try:
    import tensorflow as tf
except ImportError:
    tf = None
try:
    import tensorflow_text as tf_text
except ImportError:
    tf_text = None


@keras_hub_export("keras_hub.layers.MaskedLMMaskGenerator")
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
    masker = keras_hub.layers.MaskedLMMaskGenerator(
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

    masker = keras_hub.layers.MaskedLMMaskGenerator(
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
        _allow_python_workflow = kwargs.pop("_allow_python_workflow", True)
        super().__init__(
            _allow_python_workflow=_allow_python_workflow, **kwargs
        )

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

    @preprocessing_function
    def _call_tf(self, inputs):
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

    def _canonicalize_inputs_python(self, inputs):
        """Force inputs to a 2D list of lists."""
        if isinstance(inputs, np.ndarray):
            inputs = inputs.tolist()
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        if not inputs:
            raise ValueError(
                "At least one input is required for packing. "
                f"Received: `inputs={inputs}`"
            )

        # TODO(hongyuc): Improve the performance of `_call_python`. It becomes
        # slower when encountering large inputs compared to `_call_tf`.
        def _canonicalize_single_input(inputs):
            if isinstance(inputs, (tuple, list)):
                # Fast path for common cases:
                # If the inputs are just normal python types (or lists of
                # python types), it immediately returns.
                if not inputs:
                    return [list(inputs)], False
                first = inputs[0]
                if isinstance(
                    first, (int, str, float, bool, np.integer, np.floating)
                ):
                    return [list(inputs)], False
                if isinstance(first, (tuple, list)) and (
                    not first
                    or isinstance(
                        first[0],
                        (int, str, float, bool, np.integer, np.floating),
                    )
                ):
                    return [list(x) for x in inputs], True

                # `keras.tree.map_structure` is expensive.
                inputs = keras.tree.map_structure(convert_to_list, inputs)
                if inputs and isinstance(inputs[0], (tuple, list)):
                    return inputs, True
                else:
                    return [inputs], False
            elif tf is not None and isinstance(
                inputs, (tf.Tensor, tf.RaggedTensor)
            ):
                unbatched = inputs.shape.rank == 1
                if unbatched:
                    inputs = tf.expand_dims(inputs, 0)
                if isinstance(inputs, tf.Tensor):
                    inputs = inputs.numpy().tolist()
                else:
                    inputs = inputs.to_list()
                return inputs, not unbatched
            elif keras.ops.is_tensor(inputs):
                inputs = convert_to_list(inputs)
                if inputs and isinstance(inputs[0], (tuple, list)):
                    return inputs, True
                else:
                    return [inputs], False
            else:
                raise ValueError(
                    "Input should be a list or a list of lists. "
                    f"Received: {inputs}"
                )

        # convert_to_ragged_batch returns (x, batched) triplets.
        triplets = [_canonicalize_single_input(x) for x in inputs]
        x, batched = list(zip(*triplets))
        if len(set(batched)) != 1:
            raise ValueError(
                "All inputs for packing must have the same rank. "
                f"Received: `inputs={inputs}`."
            )
        return x[0], batched[0]

    def _call_python(self, inputs):
        inputs, is_batched = self._canonicalize_inputs_python(inputs)

        out_token_ids = []
        out_mask_positions = []
        out_mask_ids = []
        out_mask_weights = []

        for seq in inputs:
            eligible_positions = [
                i
                for i, token_id in enumerate(seq)
                if token_id not in self.unselectable_token_ids
            ]

            # Select positions
            selected_positions = [
                pos
                for pos in eligible_positions
                if random.random() < self.mask_selection_rate
            ]

            # Cap the selection if mask_selection_length is set
            if (
                self.mask_selection_length is not None
                and len(selected_positions) > self.mask_selection_length
            ):
                random.shuffle(selected_positions)
                selected_positions = selected_positions[
                    : self.mask_selection_length
                ]
                selected_positions.sort()

            seq_token_ids = list(seq)
            seq_mask_positions = []
            seq_mask_ids = []
            seq_mask_weights = []

            for pos in selected_positions:
                orig_token = seq_token_ids[pos]
                seq_mask_positions.append(pos)
                seq_mask_ids.append(orig_token)
                seq_mask_weights.append(1.0)

                # Determine replacement value
                r = random.random()
                if r < self.mask_token_rate:
                    new_token = self.mask_token_id
                elif r < self.mask_token_rate + self.random_token_rate:
                    new_token = random.randint(0, self.vocabulary_size - 1)
                else:
                    new_token = orig_token
                seq_token_ids[pos] = new_token

            # Padding if mask_selection_length is set
            if self.mask_selection_length is not None:
                padding_len = self.mask_selection_length - len(
                    selected_positions
                )
                seq_mask_positions.extend([0] * padding_len)
                seq_mask_ids.extend([0] * padding_len)
                seq_mask_weights.extend([0.0] * padding_len)

            out_token_ids.append(seq_token_ids)
            out_mask_positions.append(seq_mask_positions)
            out_mask_ids.append(seq_mask_ids)
            out_mask_weights.append(seq_mask_weights)

        if not is_batched:
            out_token_ids = out_token_ids[0]
            out_mask_positions = out_mask_positions[0]
            out_mask_ids = out_mask_ids[0]
            out_mask_weights = out_mask_weights[0]

        def _canonicalize_outputs(outputs, dtype=None):
            try:
                arr = np.array(outputs, dtype=dtype or "int32")
                if arr.dtype == object:
                    return outputs
                return arr
            except (ValueError, TypeError):
                return outputs

        outputs = {
            "token_ids": _canonicalize_outputs(out_token_ids, "int32"),
            "mask_positions": _canonicalize_outputs(
                out_mask_positions, "int32"
            ),
            "mask_ids": _canonicalize_outputs(out_mask_ids, "int32"),
            "mask_weights": _canonicalize_outputs(out_mask_weights, "float32"),
        }
        return convert_preprocessing_outputs_python(outputs)

    def call(self, inputs):
        if not self._allow_python_workflow or in_tf_function():
            return self._call_tf(inputs)
        else:
            return self._call_python(inputs)

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
