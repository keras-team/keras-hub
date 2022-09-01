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

"""Edit Distance metric."""

import tensorflow as tf
from tensorflow import keras


@keras.utils.register_keras_serializable(package="keras_nlp")
class EditDistance(keras.metrics.Metric):
    """Edit Distance metric.

    This class implements the edit distance metric, sometimes called
    Levenshtein Distance, as a `keras.metrics.Metric`. Essentially, edit
    distance is the least number of operations required to convert one string to
    another, where an operation can be one of substitution, deletion or
    insertion. By default, this metric will compute the normalized score, where
    the unnormalized edit distance score is divided by the number of tokens in
    the reference text.

    This class can be used to compute character error rate (CER) and word error
    rate (WER). You simply have to pass the appropriate tokenized text, and set
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
        - [Morris et al.](https://www.researchgate.net/publication/221478089)

    Examples:

    Various Input Types.

    Single-level Python list.
    >>> edit_distance = keras_nlp.metrics.EditDistance()
    >>> y_true = "the tiny little cat was found under the big funny bed".split()
    >>> y_pred = "the cat was found under the bed".split()
    >>> edit_distance(y_true, y_pred)
    <tf.Tensor: shape=(), dtype=float32, numpy=0.36363637>

    Rank 1 tensor.
    >>> edit_distance = keras_nlp.metrics.EditDistance()
    >>> y_true = tf.strings.split("the tiny little cat was found under the big funny bed")
    >>> y_pred = tf.strings.split("the cat was found under the bed")
    >>> edit_distance(y_true, y_pred)
    <tf.Tensor: shape=(), dtype=float32, numpy=0.36363637>

    Nested Python list.
    >>> edit_distance = keras_nlp.metrics.EditDistance()
    >>> y_true = [
    ...     "the tiny little cat was found under the big funny bed".split(),
    ...     "it is sunny today".split(),
    ... ]
    >>> y_pred = [
    ...     "the cat was found under the bed".split(),
    ...     "it is sunny but with a hint of cloud cover".split(),
    ... ]
    >>> edit_distance(y_true, y_pred)
    <tf.Tensor: shape=(), dtype=float32, numpy=0.73333335>

    Rank 2 tensor.
    >>> edit_distance = keras_nlp.metrics.EditDistance()
    >>> y_true = tf.strings.split([
    ...     "the tiny little cat was found under the big funny bed",
    ...     "it is sunny today",
    ... ])
    >>> y_pred = tf.strings.split([
    ...     "the cat was found under the bed",
    ...     "it is sunny but with a hint of cloud cover",
    ... ])
    >>> edit_distance(y_true, y_pred)
    <tf.Tensor: shape=(), dtype=float32, numpy=0.73333335>

    Pass the metric to `model.compile()`.
    >>> inputs = keras.Input(shape=(None,), dtype="string")
    >>> outputs = tf.strings.lower(inputs)
    >>> model = keras.Model(inputs, outputs)
    >>> model.compile(metrics=[keras_nlp.metrics.EditDistance()])
    >>> x = tf.strings.split(
    ...     tf.constant(
    ...         ["the tiny little cat was found under the big funny bed"]
    ...     )
    ... )
    >>> y = tf.strings.split(["the cat was found under the bed"])
    >>> y = tf.strings.split(["the cat was found under the bed"])
    >>> output = model.evaluate(y, x, return_dict=True)
    >>> output["edit_distance"]
    0.3636363744735718
    """

    def __init__(
        self,
        normalize=True,
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
            if self._aggregate_reference_length == 0:
                return 0.0
            return (
                self._aggregate_unnormalized_edit_distance
                / self._aggregate_reference_length
            )
        if self._number_of_samples == 0:
            return 0.0
        return (
            self._aggregate_unnormalized_edit_distance / self._number_of_samples
        )

    def reset_state(self):
        self._aggregate_unnormalized_edit_distance.assign(0.0)
        if self.normalize:
            self._aggregate_reference_length.assign(0.0)
        else:
            self._number_of_samples.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({"normalize": self.normalize})
        return config
