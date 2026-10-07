import keras
import numpy as np

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.utils.tensor_utils import is_float_dtype
from keras_hub.src.utils.tensor_utils import tf


def _to_nested_list(inputs):
    """Convert `inputs` to nested Python lists of tokens.

    Tensors are unwrapped at any depth, so a list of per-sample tensors works
    as well as a single batched one, and `bytes` are decoded to `str`.
    """
    if isinstance(inputs, bytes):
        return inputs.decode("utf-8", errors="ignore")
    if isinstance(inputs, (str, int, float, bool)):
        return inputs
    if tf is not None and isinstance(inputs, tf.RaggedTensor):
        inputs = inputs.to_list()
    elif tf is not None and isinstance(inputs, tf.Tensor):
        inputs = inputs.numpy().tolist()
    elif isinstance(inputs, np.ndarray):
        inputs = inputs.tolist()
    elif keras.ops.is_tensor(inputs):
        inputs = keras.ops.convert_to_numpy(inputs).tolist()
    if isinstance(inputs, (list, tuple)):
        return [_to_nested_list(x) for x in inputs]
    return inputs


def _nested_rank(inputs):
    rank = 0
    while isinstance(inputs, list):
        rank += 1
        if not inputs:
            break
        inputs = inputs[0]
    return rank


def _levenshtein(reference, hypothesis):
    """Count the edits that turn `hypothesis` into `reference`.

    An edit is a substitution, a deletion or an insertion of a single token.
    """
    # Only the previous row of the distance matrix is ever read, so the
    # quadratic table collapses to two rows of `len(hypothesis) + 1` entries.
    previous = list(range(len(hypothesis) + 1))
    for i, reference_token in enumerate(reference, start=1):
        current = [i]
        for j, hypothesis_token in enumerate(hypothesis, start=1):
            substitution = previous[j - 1] + (
                reference_token != hypothesis_token
            )
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, substitution)
            )
        previous = current
    return previous[-1]


@keras_hub_export("keras_hub.metrics.EditDistance")
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
    `y_true` and `y_pred` hold tokenized text, as nested Python lists, NumPy
    arrays, backend tensors, or `tf.Tensor`/`tf.RaggedTensor`. They are either
    a single sequence of tokens (rank 1) or a batch of them (rank 2).

    Args:
        normalize: bool. If True, the computed number of operations
            (substitutions + deletions + insertions) across all samples is
            divided by the aggregate number of tokens in all reference texts. If
            False, number of operations are calculated for every sample, and
            averaged over all the samples.
        dtype: string or tf.dtypes.Dtype. Precision of metric computation. If
            not specified, it defaults to `"float32"`.
        name: string. Name of the metric instance.
        **kwargs: Other keyword arguments.

    References:
        - [Morris et al.](https://www.researchgate.net/publication/221478089)

    Examples:

    Various Input Types.

    Single-level Python list.
    >>> edit_distance = keras_hub.metrics.EditDistance()
    >>> y_true = "the tiny little cat was found under the big funny bed".split()
    >>> y_pred = "the cat was found under the bed".split()
    >>> edit_distance(y_true, y_pred)
    <tf.Tensor: shape=(), dtype=float32, numpy=0.36363637>

    Nested Python list.
    >>> edit_distance = keras_hub.metrics.EditDistance()
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
    """

    def __init__(
        self,
        normalize=True,
        dtype="float32",
        name="edit_distance",
        **kwargs,
    ):
        super().__init__(name=name, dtype=dtype, **kwargs)

        if not is_float_dtype(dtype):
            raise ValueError(
                "`dtype` must be a floating point type. "
                f"Received: dtype={dtype}"
            )

        self.normalize = normalize

        self._aggregate_unnormalized_edit_distance = self.add_weight(
            shape=(),
            initializer="zeros",
            dtype=self.dtype,
            name="aggregate_unnormalized_edit_distance",
        )
        if normalize:
            self._aggregate_reference_length = self.add_weight(
                shape=(),
                initializer="zeros",
                dtype=self.dtype,
                name="aggregate_reference_length",
            )
        else:
            self._number_of_samples = self.add_weight(
                shape=(),
                initializer="zeros",
                dtype=self.dtype,
                name="number_of_samples",
            )

    def update_state(self, y_true, y_pred, sample_weight=None):
        def validate_and_fix_rank(inputs, tensor_name):
            inputs = _to_nested_list(inputs)
            rank = _nested_rank(inputs)
            if rank == 1:
                return [inputs]
            elif rank == 2:
                return inputs
            else:
                raise ValueError(
                    f"{tensor_name} must be of rank 1 or 2. Found rank: {rank}"
                )

        y_true = validate_and_fix_rank(y_true, "y_true")
        y_pred = validate_and_fix_rank(y_pred, "y_pred")

        if len(y_true) != len(y_pred):
            raise ValueError(
                "y_true and y_pred must have the same number of samples. "
                f"Received: len(y_true)={len(y_true)}, "
                f"len(y_pred)={len(y_pred)}"
            )

        if self.normalize:
            reference_length = sum(len(reference) for reference in y_true)
            self._aggregate_reference_length.assign_add(
                np.array(reference_length, dtype=self.dtype)
            )

        edit_distance = sum(
            _levenshtein(reference, hypothesis)
            for reference, hypothesis in zip(y_true, y_pred)
        )
        self._aggregate_unnormalized_edit_distance.assign_add(
            np.array(edit_distance, dtype=self.dtype)
        )
        if not self.normalize:
            self._number_of_samples.assign_add(
                np.array(len(y_true), dtype=self.dtype)
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
