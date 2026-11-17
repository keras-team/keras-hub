"""Input canonicalization helpers for the text metrics.

The text metrics (`Bleu`, `RougeN`, `RougeL`) accept Python strings, nested
lists of strings, NumPy arrays, backend tensors, and TensorFlow dense or
ragged tensors. These helpers turn all of those into nested Python lists of
`str` without importing TensorFlow, so the metrics can run when it is not
installed.
"""

import numpy as np

from keras_hub.src.utils.tensor_utils import convert_to_numpy


def _to_str(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    if isinstance(x, np.str_):
        return str(x)
    return x


def to_nested_list(inputs):
    """Convert string inputs to (possibly nested) Python lists of `str`.

    Accepts a `str` or `bytes` scalar, a (possibly ragged) nested list or
    tuple, a NumPy array, a backend tensor, or a TensorFlow `tf.Tensor` or
    `tf.RaggedTensor`. A scalar input is returned as a `str`.
    """
    if isinstance(inputs, (str, bytes, np.str_)):
        return _to_str(inputs)
    if isinstance(inputs, (list, tuple)):
        return [to_nested_list(x) for x in inputs]
    # A `tf.RaggedTensor` can't be converted to a NumPy array directly, but
    # it knows how to turn itself into nested lists. Duck typing keeps
    # TensorFlow out of this module.
    if hasattr(inputs, "to_list") and hasattr(inputs, "ragged_rank"):
        return to_nested_list(inputs.to_list())
    inputs = convert_to_numpy(inputs)
    if inputs.ndim == 0:
        return _to_str(inputs.item())
    return to_nested_list(inputs.tolist())


def rank_of(inputs):
    """Rank of a nested list of strings. A `str` has rank 0."""
    rank = 0
    while isinstance(inputs, list):
        rank += 1
        if not inputs:
            break
        inputs = inputs[0]
    return rank


def validate_and_fix_rank(inputs, tensor_name, base_rank=0):
    """Canonicalize `inputs` to rank `base_rank + 1`.

    A rank `base_rank` input is treated as a single unbatched sample and
    wrapped in a batch dimension. A rank `base_rank + 2` input is allowed
    only when its innermost dimension has size 1, in which case that
    dimension is squeezed. Anything else raises a `ValueError`.
    """
    inputs = to_nested_list(inputs)
    rank = rank_of(inputs)
    if rank == base_rank:
        return [inputs]
    if rank == base_rank + 1:
        return inputs
    if rank == base_rank + 2:
        return _squeeze_last(inputs, tensor_name, rank)
    raise ValueError(
        f"{tensor_name} must be of rank {base_rank}, {base_rank + 1}, or "
        f"{base_rank + 2}. Found rank: {rank}"
    )


def _squeeze_last(inputs, tensor_name, rank):
    """Drop a trailing dimension of size 1 from a nested list."""
    if isinstance(inputs[0], list) and (
        not inputs[0] or isinstance(inputs[0][0], list)
    ):
        # Not yet at the innermost dimension; recurse.
        return [_squeeze_last(x, tensor_name, rank) for x in inputs]
    if any(len(x) != 1 for x in inputs):
        raise ValueError(
            f"{tensor_name} is of rank {rank}. The last dimension must be of "
            "size 1."
        )
    return [x[0] for x in inputs]
