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

"""Edit Distance metric implementation based on `keras.metrics.Metric`."""

import tensorflow as tf
from tensorflow import keras


class EditDistance(keras.metrics.Metric):
    """Edit Distance.

    This class implements the edit distance metric, sometimes called
    Levenshtein Distance. Essentially, edit distance is the least number of
    operations required to convert one string to another, where a operation
    can be one of substitution, deletion or insertion. In the normalized version,
    the above score is divided by the number of tokens in the reference text.

    This class can be used to compute character error rate (CER)/word error rate
    (WER). You simply have to pass the appropriate tokenized text, and set
    `normalize` to True.

    Note on input shapes:
    `y_true` and `y_pred` can either be tensors of rank 1 or ragged tensors of
    rank 2. These tensors contain tokenized text.

    Args:
        normalize: bool. If True, the computed number of operations
            (substitutions + deletions + insertions) across all samples is
            divided by the aggregate number of tokens in all reference texts. If
            False, number of operations are calculated for every sample, and
            averaged over all the samples.
        dtype: string or tf.dtypes.Dtype. Precision of metric computation. If
            not specified, it defaults to tf.float32.
        name: string. Name of the metric instance.
        **kwargs: Other keyword arguments.

    References:
        - [Morris et al.](https://www.researchgate.net/publication/221478089_From_WER_and_RIL_to_MER_and_WIL_improved_evaluation_measures_for_connected_speech_recognition)
    """

    def __init__(
        self,
        normalize=False,
        dtype=None,
        name="edit_distance",
        **kwargs,
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)

        if not tf.as_dtype(self.dtype).is_floating:
            raise ValueError(
                "`dtype` must be a floating point type. "
                f"Received: dtype={dtype}"
            )

        self.normalize = normalize

        self._aggregate_unnormalized_edit_distance = self.add_weight(
            name="aggregate_unnormalized_edit_distance",
            initializer="zeros",
            dtype=self.dtype,
        )
        if normalize:
            self._aggregate_reference_length = self.add_weight(
                name="aggregate_reference_length",
                initializer="zeros",
                dtype=self.dtype,
            )
        else:
            self._number_of_samples = self.add_weight(
                name="number_of_samples",
                initializer="zeros",
                dtype=self.dtype,
            )

    def update_state(self, y_true, y_pred, sample_weight=None):
        def validate_and_fix_rank(inputs, tensor_name):
            if not isinstance(inputs, (tf.Tensor, tf.RaggedTensor)):
                inputs = tf.ragged.constant(inputs)

            if inputs.shape.rank == 1:
                return tf.RaggedTensor.from_tensor(inputs[tf.newaxis])
            elif inputs.shape.rank == 2:
                return inputs
            else:
                raise ValueError(
                    f"{tensor_name} must be of rank 1 or 2. "
                    f"Found rank: {inputs.shape.rank}"
                )

        y_true = validate_and_fix_rank(y_true, "y_true")
        y_pred = validate_and_fix_rank(y_pred, "y_pred")

        if self.normalize:
            self._aggregate_reference_length.assign_add(
                tf.cast(tf.size(y_true.flat_values), dtype=self.dtype)
            )

        def calculate_edit_distance(args):
            reference, hypothesis = args

            reference = tf.sparse.from_dense([reference])
            hypothesis = tf.sparse.from_dense([hypothesis])

            edit_distance = tf.squeeze(
                tf.edit_distance(
                    hypothesis=hypothesis,
                    truth=reference,
                    normalize=False,
                )
            )

            self._aggregate_unnormalized_edit_distance.assign_add(
                tf.cast(edit_distance, dtype=self.dtype)
            )
            if not self.normalize:
                self._number_of_samples.assign_add(tf.cast(1, dtype=self.dtype))
            return 0

        _ = tf.map_fn(
            fn=calculate_edit_distance,
            elems=(y_true, y_pred),
            fn_output_signature=tf.int8,
        )

    def result(self):
        if self.normalize:
            return (
                self._aggregate_unnormalized_edit_distance
                / self._aggregate_reference_length
            )
        return (
            self._aggregate_unnormalized_edit_distance / self._number_of_samples
        )

    def reset_state(self):
        if self.normalize:
            self._aggregate_unnormalized_edit_distance.assign(0.0)
            self._aggregate_reference_length.assign(0.0)
        else:
            self._aggregate_unnormalized_edit_distance.assign(0.0)
            self._number_of_samples.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({"normalize": self.normalize})
        return config
