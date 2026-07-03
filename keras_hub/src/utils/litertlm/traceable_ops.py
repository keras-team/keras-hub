"""Traceable replacements for Keras torch-backend ops used during export.

``torch.export`` (used by ``litert_torch`` to trace KerasHub models for
LiteRT-LM) cannot lower every op that Keras's torch backend produces --
some introduce fused ATen ops, runtime assertions, or unbacked symbolic
shapes that ``litert_torch`` cannot translate to TFLite. This module holds
self-contained, drop-in replacements for the handful of ops that need this
treatment (``one_hot``, ``repeat``, ``slice``, ``take``, ``scatter_update``,
``dot_product_attention``, ``arange``), plus context managers that
temporarily monkeypatch Keras's torch backend to use them. This has no
relationship to ``KerasHubLiteRTAdapter`` beyond being a dependency used
while tracing it -- it is a standalone "make these ops traceable" shim
library.
"""

import contextlib
import unittest.mock

import torch
from keras.src import backend
from keras.src.backend.common import dtypes as keras_dtypes
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


def _normalize_start_indices(start_indices):
    """Convert ``start_indices`` to a list preserving tensor elements."""
    if isinstance(start_indices, (list, tuple)):
        return list(start_indices)
    start_indices = torch_core.convert_to_tensor(start_indices, dtype="int64")
    if start_indices.ndim != 1:
        raise ValueError(
            "`start_indices` must be a 1-D tensor or a list/tuple of ints. "
            f"Received shape: {tuple(start_indices.shape)}."
        )
    return list(start_indices.reshape(-1).unbind())


def _make_patched_slice():
    """Return a traceable ``slice`` replacement."""

    def _patched_slice(inputs, start_indices, shape):
        inputs = torch_core.convert_to_tensor(inputs)

        starts = _normalize_start_indices(start_indices)

        if isinstance(shape, (list, tuple)):
            lengths = list(shape)
        else:
            shape = torch_core.convert_to_tensor(shape, dtype="int64")
            lengths = list(shape.reshape(-1).unbind())

        def _is_dynamic(value):
            # ``torch.SymInt`` values are not plain Python ints and require
            # tensor-based slicing to avoid data-dependent guards.
            return isinstance(value, torch.Tensor) or isinstance(
                value, torch.SymInt
            )

        # Dimensions whose start or length is dynamic.
        dynamic_dims = [
            dim
            for dim, (start, length) in enumerate(zip(starts, lengths))
            if _is_dynamic(start) or _is_dynamic(length)
        ]

        # No dynamic values → use Python slice objects directly.
        if len(dynamic_dims) == 0:
            slices = tuple(
                slice(start, start + length)
                for start, length in zip(starts, lengths)
            )
            return inputs[slices]

        # Single dynamic dimension → build indices with ``torch.arange`` and
        # use ``index_select``. This keeps the output shape symbolic and avoids
        # unbacked symbols that ``torch.export`` cannot resolve.
        if len(dynamic_dims) == 1:
            dim = dynamic_dims[0]
            start = starts[dim]
            if not isinstance(start, torch.Tensor):
                start = torch_core.convert_to_tensor(
                    start, dtype="int32", device=inputs.device
                )
            start = start.reshape(())
            length = lengths[dim]

            indices = torch.arange(
                length, dtype=torch.int32, device=inputs.device
            )
            indices = indices + start
            result = torch.index_select(inputs, dim, indices)

            # Apply static slicing for the remaining dimensions.
            for d, (s, l) in enumerate(zip(starts, lengths)):
                if d != dim and (s != 0 or l != result.shape[d]):
                    result = torch.narrow(result, d, s, l)
            return result

        # Multiple dynamic dimensions are not supported for LiteRT-LM export
        # because ``torch.export`` cannot resolve the resulting unbacked
        # symbols. Fail fast with an actionable message instead of falling
        # back to the original implementation and producing a cryptic export
        # error.
        raise NotImplementedError(
            "Slicing with multiple dynamic dimensions is not supported for "
            "LiteRT-LM export. Received dynamic dims "
            f"{dynamic_dims}. Consider materializing start/length values as "
            "static ints or simplifying the slice operation."
        )

    return _patched_slice


@contextlib.contextmanager
def _traceable_slice_scope():
    """Temporarily patch Keras torch-backend ``slice`` for torch.export.

    Keras ``ops.slice`` can introduce unbacked symbols during
    ``torch.export.export`` for dynamic start/length values. This patch keeps
    slicing traceable for the common single-dynamic-dimension case.

    Uses ``unittest.mock.patch.object`` so restoration is reliable even when
    an exception escapes.
    """
    with unittest.mock.patch.object(
        torch_core,
        "slice",
        _make_patched_slice(),
    ):
        yield


def _traceable_dot_product_attention(
    query,
    key,
    value,
    bias=None,
    mask=None,
    scale=None,
    is_causal=False,
    flash_attention=None,
    attn_logits_soft_cap=None,
):
    """Traceable replacement for Keras torch-backend ``dot_product_attention``.

    ``torch.nn.functional.scaled_dot_product_attention`` lowers to fused ATen
    ops such as ``aten._scaled_dot_product_efficient_attention`` that
    ``litert_torch`` cannot translate to TFLite.  This implementation expands
    attention to a plain ``matmul`` + ``softmax`` + ``matmul`` sequence that
    ``litert_torch`` handles well.

    The function mirrors Keras's ``dot_product_attention`` signature and shape
    convention: inputs are ``[batch, seq_len, num_heads, head_dim]`` and the
    output is returned in the same layout.

    Unlike the original Keras torch-backend op, this implementation does
    **not** internally broadcast mismatched query/key-value head counts for
    grouped-query attention. Every current KerasHub attention layer that
    calls ``dot_product_attention`` (e.g. ``LlamaAttention``,
    ``GemmaAttention``) already repeats the key/value heads up to the query
    head count before calling it, so this is not a behavior change in
    practice -- but a caller relying on the original op's implicit GQA
    broadcast would get silently wrong (shape-broadcast) results here.
    """
    del flash_attention  # Fused flash attention is not exportable.

    query = torch_core.convert_to_tensor(query)
    key = torch_core.convert_to_tensor(key)
    value = torch_core.convert_to_tensor(value)

    compute_dtype = backend.result_type(query.dtype, key.dtype, value.dtype)
    query = torch_core.cast(query, compute_dtype)
    key = torch_core.cast(key, compute_dtype)
    value = torch_core.cast(value, compute_dtype)

    if scale is None:
        scale = float(query.shape[-1]) ** -0.5
    scale = torch_core.convert_to_tensor(scale, dtype=compute_dtype)

    if mask is not None:
        mask = torch_core.convert_to_tensor(mask, dtype="bool")
        if is_causal:
            q_len, kv_len = query.shape[1], key.shape[1]
            causal_mask = torch.tril(
                torch.ones(
                    (q_len, kv_len), dtype=torch.bool, device=mask.device
                )
            )
            mask = torch.logical_and(mask, causal_mask)
        is_causal = False
    elif is_causal:
        q_len, kv_len = query.shape[1], key.shape[1]
        mask = torch.tril(
            torch.ones((q_len, kv_len), dtype=torch.bool, device=query.device)
        )

    # Move heads to the batch dimension to match SDPA's score layout
    # [batch, num_heads, seq_len, head_dim].
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    if bias is not None:
        scores = scores + torch_core.convert_to_tensor(
            bias, dtype=compute_dtype
        )

    if mask is not None:
        large_neg = torch.tensor(
            torch.finfo(scores.dtype).min,
            dtype=scores.dtype,
            device=scores.device,
        )
        scores = torch.where(mask, scores, large_neg)

    if attn_logits_soft_cap is not None:
        cap = torch_core.convert_to_tensor(
            attn_logits_soft_cap, dtype=compute_dtype
        )
        scores = torch.tanh(scores / cap) * cap

    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn, value)
    return output.transpose(1, 2)


_traceable_dot_product_attention_scope = _make_scope(
    torch_backend_nn, "dot_product_attention", _traceable_dot_product_attention
)


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
        axis = dims + axis
    if axis < 0 or axis >= dims:
        raise ValueError(
            f"`axis` {axis} is out of bounds for one-hot output with "
            f"{dims} dimensions."
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


# Capture the original ``arange`` at module load so the patched version can
# delegate to it without closing over a local variable.
_ORIGINAL_ARANGE = torch_backend_numpy.arange


def _patched_arange(start, stop=None, step=None, dtype=None):
    """Patch Keras torch-backend ``arange`` to default integer ranges to int32.

    ``torch.arange`` returns int64 for integer arguments. ``litert_torch``'s
    i64-to-i32 conversion pass does not always propagate through nested
    ``func.call`` boundaries, so integer ranges produced inside the model
    (e.g. position-embedding indices) must be int32 from the start.
    """
    if dtype is None:
        dtypes_to_resolve = [
            getattr(start, "dtype", type(start)),
        ]
        if stop is not None:
            dtypes_to_resolve.append(getattr(stop, "dtype", type(stop)))
        if step is not None:
            dtypes_to_resolve.append(getattr(step, "dtype", type(step)))
        resolved = keras_dtypes.result_type(*dtypes_to_resolve)
        if str(resolved).startswith("int"):
            dtype = torch.int32
    return _ORIGINAL_ARANGE(start, stop=stop, step=step, dtype=dtype)


_traceable_arange_scope = _make_scope(
    torch_backend_numpy, "arange", _patched_arange
)


def _patched_take(x, indices, axis=None):
    """Patch Keras torch-backend ``take`` to keep embedding indices as int32.

    The default implementation casts integer indices to int64 before calling
    ``torch.nn.functional.embedding``. ``litert_torch``'s TFLite embedding
    lowering expects int32 indices consistent with the traced function
    signature, so we keep indices in int32 for the embedding-lookup case.
    """
    x = torch_core.convert_to_tensor(x)
    indices = torch_core.convert_to_tensor(indices, dtype=torch.int32)
    x_dim = x.shape[axis] if axis is not None else x.shape[0]
    indices = torch.where(
        indices < 0,
        indices + x_dim,
        indices,
    )
    if x.ndim == 2 and axis == 0:
        return torch.nn.functional.embedding(indices, x)
    if axis is None:
        x = torch.reshape(x, (-1,))
        axis = 0
    axis = torch_backend_numpy.canonicalize_axis(axis, x.ndim)
    shape = x.shape[:axis] + indices.shape + x.shape[axis + 1 :]
    indices = indices.ravel()
    out = torch.index_select(x, dim=axis, index=indices).squeeze(axis)
    return out.reshape(shape)


_traceable_take_scope = _make_scope(torch_backend_numpy, "take", _patched_take)


_SCATTER_UPDATE_REDUCTION_OPS = {
    "max": torch.maximum,
    "min": torch.minimum,
    "mul": lambda a, b: a * b,
}


def _patched_scatter_update(inputs, indices, updates, reduction=None):
    """Patch Keras torch-backend ``scatter_update`` to keep indices int32.

    The default implementation casts indices to int64. ``litert_torch``'s
    TFLite scatter lowering expects int32 indices, so we keep them in int32
    during export.
    """
    inputs = torch_core.convert_to_tensor(inputs)
    indices = torch_core.convert_to_tensor(indices, dtype=torch.int32)
    updates = torch_core.convert_to_tensor(updates, dtype=inputs.dtype)
    indices = torch.transpose(indices, 0, 1)
    idx = tuple(indices)

    outputs = torch.clone(inputs)
    if reduction is None:
        outputs[idx] = updates
    elif reduction == "add":
        outputs.index_put_(idx, updates, accumulate=True)
    elif reduction in _SCATTER_UPDATE_REDUCTION_OPS:
        op_fn = _SCATTER_UPDATE_REDUCTION_OPS[reduction]
        indices_t = indices.T
        for i in range(indices_t.shape[0]):
            idx_i = tuple(indices_t[i])
            outputs[idx_i] = op_fn(outputs[idx_i], updates[i])
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")
    return outputs


_traceable_scatter_update_scope = _make_scope(
    torch_core, "scatter_update", _patched_scatter_update
)


def _is_scalar_integer(value):
    """Return ``True`` if *value* is a scalar integer (Python or numpy)."""
    if isinstance(value, int) and not isinstance(value, bool):
        return True
    # Accept 0-D numpy integer arrays / tensors with a single integer value.
    if hasattr(value, "dtype") and hasattr(value, "ndim"):
        return value.ndim == 0 and "int" in str(value.dtype)
    return False


def _traceable_repeat(x, repeats, axis=None):
    """Traceable replacement for Keras torch-backend ``numpy.repeat``.

    ``torch.repeat_interleave`` lowers to ``aten.repeat_interleave.Tensor``,
    which ``litert_torch`` cannot translate to TFLite. For the common case of
    repeating a tensor by a scalar integer along a single axis (e.g. GQA
    key/value head repetition), this implementation uses
    ``unsqueeze + expand + reshape``, which ``litert_torch`` handles.
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

    # Fall back to the original implementation for list/tuple repeats or
    # dynamic repeat counts.
    return torch_backend_numpy.repeat(x, repeats, axis=axis)


_traceable_repeat_scope = _make_scope(
    torch_backend_numpy, "repeat", _traceable_repeat
)


@contextlib.contextmanager
def traceable_ops_scope():
    """Enter every traceable-op patch scope at once.

    Combines the seven individual patch scopes (slice, dot_product_attention,
    one_hot, repeat, arange, take, scatter_update) into a single context
    manager via ``contextlib.ExitStack``, so callers open one scope instead of
    nesting seven ``with`` statements.
    """
    with contextlib.ExitStack() as stack:
        stack.enter_context(_traceable_slice_scope())
        stack.enter_context(_traceable_dot_product_attention_scope())
        stack.enter_context(_traceable_one_hot_scope())
        stack.enter_context(_traceable_repeat_scope())
        stack.enter_context(_traceable_arange_scope())
        stack.enter_context(_traceable_take_scope())
        stack.enter_context(_traceable_scatter_update_scope())
        yield
