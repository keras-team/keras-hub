"""Traceable replacements for Keras torch-backend ops used during export.

``torch.export`` (used by ``litert_torch`` to trace KerasHub models for
LiteRT-LM) cannot lower every op that Keras's torch backend produces --
some introduce fused ATen ops, runtime assertions, or unbacked symbolic
shapes that ``litert_torch`` cannot translate to TFLite. This module holds
self-contained, drop-in replacements for the handful of ops that need this
treatment (``one_hot``, ``repeat``, ``slice``, ``take``, ``scatter_update``,
``dot_product_attention``, ``amax``), plus context managers that
temporarily monkeypatch Keras's torch backend to use them.

These replacements live in keras-hub rather than upstream in Keras because
they are export-specific workarounds for ``torch.export`` and LiteRT
lowering limitations, not improvements to the torch backend generally.
"""

import contextlib
import unittest.mock

import numpy as np
import torch
from keras.src import backend
from keras.src.backend.torch import core as torch_core
from keras.src.backend.torch import nn as torch_backend_nn
from keras.src.backend.torch import numpy as torch_backend_numpy


def _make_scope(module, attr, replacement):
    """Build a context manager that patches ``module.attr`` to *replacement*."""

    @contextlib.contextmanager
    def _scope():
        with unittest.mock.patch.object(module, attr, replacement):
            yield

    return _scope


def _patched_one_hot(x, num_classes, axis=-1, dtype=None, sparse=False):
    """Traceable replacement for Keras torch-backend ``one_hot``.

    ``torch.nn.functional.one_hot`` inserts runtime assertions that class
    values are non-negative. Under ``torch.export`` these become
    ``aten._assert_async.msg`` ops, which ``litert_torch`` cannot lower.

    This implementation uses equality against ``torch.arange``, which produces
    the same result for non-negative indices and does not introduce
    unlowerable assertions. Negative indices are preserved as all-zero vectors,
    matching the original behavior.

    Integer tensors are kept in int32 so the exported MLIR remains compatible
    with ``litert_torch``'s i32-based TFLite lowering.
    """
    if sparse:
        raise ValueError("Unsupported value `sparse=True` with torch backend")
    x = torch_core.convert_to_tensor(x, dtype=torch.int32)
    x_clamped = torch.clamp(x, min=0)
    output = x_clamped.unsqueeze(-1) == torch.arange(
        num_classes, dtype=torch.int32, device=x.device
    )
    # Preserve original behavior for negative indices.
    zero = torch.zeros_like(output)
    output = torch.where(x.unsqueeze(-1) >= 0, output, zero)
    if dtype is None:
        dtype = "float32"
    output = torch_core.convert_to_tensor(output, dtype=dtype)
    dims = output.dim()
    if axis < 0:
        original_axis = axis
        axis = dims + axis
    else:
        original_axis = axis
    if axis < 0 or axis >= dims:
        raise ValueError(
            "`axis` is out of bounds for one-hot output with "
            f"{dims} dimensions. Received: axis={original_axis}."
        )
    if axis != dims - 1:
        new_axes_order = list(range(dims))
        new_axes_order[axis] = dims - 1
        for ax in range(axis + 1, dims):
            new_axes_order[ax] -= 1
        output = output.permute(new_axes_order)
    return output


_traceable_one_hot_scope = _make_scope(
    torch_backend_nn, "one_hot", _patched_one_hot
)


def _is_scalar_integer(value):
    """Return ``True`` if *value* is a scalar integer (Python or numpy)."""
    if isinstance(value, int) and not isinstance(value, bool):
        return True
    # Accept 0-D numpy integer arrays / tensors with a single integer value.
    if hasattr(value, "dtype") and hasattr(value, "ndim"):
        return value.ndim == 0 and "int" in str(value.dtype)
    return False


# Capture the original ``repeat`` at module load so the patch can delegate
# to it for every input form outside the scalar-tensor intercept.
_ORIGINAL_REPEAT = torch_backend_numpy.repeat


def _traceable_repeat(x, repeats, axis=None):
    """Intercept 0-D tensor ``repeats`` (and ``repeats == 1``) with an axis.

    Keras's own ``repeat`` fast-paths only plain Python ints; every other
    input form defers to the original implementation unchanged.
    """
    x = torch_core.convert_to_tensor(x)

    if axis is not None and _is_scalar_integer(repeats):
        repeats = int(repeats)
        if repeats < 0:
            raise ValueError("`repeats` must be non-negative.")
        if repeats == 1:
            return x
        if axis < 0:
            axis = x.ndim + axis
        shape = list(x.shape)
        x = x.unsqueeze(axis + 1)
        expand_shape = [-1] * x.ndim
        expand_shape[axis + 1] = repeats
        x = x.expand(expand_shape)
        new_shape = list(shape)
        new_shape[axis] = shape[axis] * repeats
        return x.reshape(new_shape)

    return _ORIGINAL_REPEAT(x, repeats, axis=axis)


_traceable_repeat_scope = _make_scope(
    torch_backend_numpy, "repeat", _traceable_repeat
)


# Capture the original ``amax`` at module load so the patched version can
# delegate to it for every input form that does not trigger the layout bug.
_ORIGINAL_AMAX = torch_backend_numpy.amax


def _patched_amax(x, axis=None, keepdims=False):
    """Traceable replacement for Keras torch-backend ``amax``.

    ``keras.ops.max`` / ``keras.ops.amax`` with an integer ``axis`` lower to
    ``aten.amax``. ``litert_torch``'s layout-optimization pass registers a
    *checker* for ``aten.amax`` that forces the op to NHWC whenever its input
    is 4-D (``layout_check.py``), but registers **no matching NHWC rewriter**
    (``layout_rewrite.py``), so a 4-D ``aten.amax`` aborts conversion with
    ``RuntimeError: NHWC node rewriter not found: amax``. GPT-OSS hits this in
    its attention-sink softmax stabilization
    (``gpt_oss_attention.py``: ``ops.max(combined_logits, axis=-1,
    keepdims=True)`` on a ``[batch, heads, q, k]`` tensor); the litert-torch
    gap is https://github.com/google-ai-edge/litert-torch/issues/1126.

    For the single-integer-axis reduction of a 4-D tensor -- the only case that
    trips the missing rewriter -- this routes through ``torch.max(dim=...)``,
    which lowers to ``aten.max.dim`` (a *registered* rewriter). That is the
    identical reduction and yields bit-identical values. Every other input form
    (other ranks, tuple axes, ``axis=None``) defers to the original ``amax``
    unchanged, so this is a no-op transform outside the triggering case.
    """
    x = torch_core.convert_to_tensor(x)
    # NumPy integer axes (e.g. ``np.int64`` from shape arithmetic) are not
    # Python ``int``; normalize so they take the same traceable path.
    if isinstance(axis, np.integer):
        axis = int(axis)
    if axis is not None and isinstance(axis, int) and x.ndim == 4:
        return torch.max(x, dim=axis, keepdim=keepdims).values
    return _ORIGINAL_AMAX(x, axis=axis, keepdims=keepdims)


_traceable_amax_scope = _make_scope(torch_backend_numpy, "amax", _patched_amax)


@contextlib.contextmanager

def traceable_ops_scope():
    """Enter a context where Keras PyTorch backend ops are replaced with export-traceable shims."""
    return _TraceableOpsScope(
        _traceable_one_hot_scope,
        _traceable_repeat_scope,
        _traceable_amax_scope,
    )

class _TraceableOpsScope:
    def __init__(self, *scopes):
        self.scopes = scopes
        self._entered = []
    def __enter__(self):
        for s in self.scopes:
            s.__enter__()
            self._entered.append(s)
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        while self._entered:
            s = self._entered.pop()
            s.__exit__(exc_type, exc_val, exc_tb)
