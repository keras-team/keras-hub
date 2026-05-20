"""PyTorch adapter modules for exporting KerasHub CausalLM models to
LiteRT-LM."""

import contextlib

import torch
from keras.src.backend.torch import core as torch_core
from torch import nn


class KerasHubLiteRTAdapter(nn.Module):
    """Adapter that wraps a KerasHub CausalLM for LiteRT-LM export.

    The adapter exposes `forward_prefill` and `forward_decode` signatures
    compatible with `litert_torch.signature(...)`:

        inputs:
            tokens:       int32 [batch, seq_len]
            input_pos:    int32 [seq_len]   (position indices)
            mask:         optional float mask (ignored by most models)
            kv_cache_k_0, kv_cache_v_0, ...: per-layer KV caches

        outputs (prefill):
            kv_cache_k_0, kv_cache_v_0, ...: updated per-layer KV caches
            (no logits – LiteRT-LM extracts last-token logits via decode)

        outputs (decode):
            logits:       float [batch, seq_len, vocab_size]
            kv_cache_k_0, kv_cache_v_0, ...: updated per-layer KV caches

    The adapter stacks per-layer k/v tensors into the Keras cache format
    (``[batch, num_layers, 2, cache_length, num_kv_heads, head_dim]``),
    calls ``model.call_with_cache()``, and unstacks the result.
    """

    def __init__(self, keras_model, num_layers, cache_length):
        super().__init__()
        self.keras_model = keras_model
        self.num_layers = num_layers
        self.cache_length = cache_length

    def forward_prefill(self, tokens, input_pos, mask=None, **kv_cache):
        """Prefill step – processes the full prompt at the given cache position.

        LiteRT-LM requires prefill to return **only** KV cache tensors
        (no logits).  The runtime extracts the last-token logits internally
        via a dedicated decode step.

        ``input_pos`` is a 1-D int64 tensor (e.g. ``[0, 1, 2, ...]`` for the
        first turn, or ``[N, N+1, ...]`` for subsequent turns).  The first
        element is used as the cache-update index so that prefill appends to
        the existing cache instead of overwriting from position 0.
        """
        cache = self._stack_kv_cache(kv_cache)
        # The first element of input_pos is the start position.
        cache_update_index = input_pos[0]
        logits, _, updated_cache = self.keras_model.call_with_cache(
            tokens, cache, cache_update_index
        )
        # Return ONLY the KV cache outputs (no logits).
        outputs = self._unstack_kv_cache(updated_cache)
        return outputs

    def forward_decode(self, tokens, input_pos, mask=None, **kv_cache):
        """Decode step – processes a single token at *input_pos*.

        ``input_pos`` is a scalar int32 tensor (e.g. ``[3]``).  It is passed
        directly as the cache-update index so that the value remains a tensor
        inside the exported graph and is not baked in as a Python constant.
        """
        cache = self._stack_kv_cache(kv_cache)
        # Squeeze to a 0-D tensor so Keras cache operations receive a scalar.
        cache_update_index = input_pos.reshape(())
        logits, _, updated_cache = self.keras_model.call_with_cache(
            tokens, cache, cache_update_index
        )
        outputs = self._unstack_kv_cache(updated_cache)
        outputs["logits"] = logits
        return outputs

    def _stack_kv_cache(self, kv_cache):
        """Stack flat ``kv_cache_k_N`` / ``kv_cache_v_N`` into Keras format."""
        k_list = [kv_cache[f"kv_cache_k_{i}"] for i in range(self.num_layers)]
        v_list = [kv_cache[f"kv_cache_v_{i}"] for i in range(self.num_layers)]
        k_stack = torch.stack(k_list, dim=1)
        v_stack = torch.stack(v_list, dim=1)
        return torch.stack([k_stack, v_stack], dim=2)

    def _unstack_kv_cache(self, cache):
        """Split Keras cache back into per-layer output tensors."""
        outputs = {}
        for i in range(self.num_layers):
            outputs[f"kv_cache_k_{i}"] = cache[:, i, 0, ...]
            outputs[f"kv_cache_v_{i}"] = cache[:, i, 1, ...]
        return outputs


@contextlib.contextmanager
def _traceable_slice_update_scope():
    """Temporarily patch Keras torch-backend ``slice_update`` for torch.export.

    Keras ``ops.slice_update`` converts tensor ``start_indices`` to Python
    ints via ``.tolist()``, which fails during ``torch.export.export`` when
    the index is a dynamic tensor (e.g. the decode position).

    This patch replaces the implementation entirely for the export window:
    * Tensor starts (dynamic) with ``update_len == 1`` → ``index_copy_``.
    * All-Python-int starts (constant) → direct slice assignment without
      calling ``.item()`` or the original Keras logic.
    """
    original_slice_update = torch_core.slice_update
    original_slice = torch_core.slice

    def _patched_slice_update(inputs, start_indices, updates):
        inputs = torch_core.convert_to_tensor(inputs)
        updates = torch_core.convert_to_tensor(updates)

        # Normalise start_indices to a list while preserving types.
        if isinstance(start_indices, (list, tuple)):
            starts = list(start_indices)
        else:
            start_indices = torch_core.convert_to_tensor(
                start_indices, dtype="int64"
            )
            starts = [start_indices[i] for i in range(start_indices.numel())]

        # Dimensions whose start is a tensor → potentially dynamic.
        tensor_dims = []
        for dim, start in enumerate(starts):
            if isinstance(start, torch.Tensor):
                tensor_dims.append(dim)

        # Single dynamic dimension → loop over index_copy_ (works for any
        # update_len because update_len is a constant at export time).
        if len(tensor_dims) == 1:
            dim = tensor_dims[0]
            update_len = updates.shape[dim]
            start = starts[dim].reshape(())

            outputs = torch.clone(inputs)
            for i in range(update_len):
                index = (start + i).reshape(()).unsqueeze(0)
                update_slice = updates.select(dim, i).unsqueeze(dim)
                outputs.index_copy_(dim, index, update_slice)
            return outputs

        if len(tensor_dims) > 1:
            raise RuntimeError(
                "slice_update patch does not support multiple dynamic start "
                f"indices. Dynamic dims: {tensor_dims}."
            )

        # All starts are plain Python ints → direct slice assignment.
        # We avoid the original Keras implementation because it converts
        # the starts to a tensor and calls ``.item()``, which breaks under
        # ``torch.export`` on older Keras versions (< 3.15).
        outputs = torch.clone(inputs)
        slices = []
        for start, size in zip(starts, updates.shape):
            slices.append(slice(start, start + size))
        outputs[slices] = updates
        return outputs

    def _patched_slice(inputs, start_indices, shape):
        inputs = torch_core.convert_to_tensor(inputs)

        # Normalise start_indices and shape to lists while preserving types.
        if isinstance(start_indices, (list, tuple)):
            starts = list(start_indices)
        else:
            start_indices = torch_core.convert_to_tensor(
                start_indices, dtype="int64"
            )
            starts = [start_indices[i] for i in range(start_indices.numel())]

        if isinstance(shape, (list, tuple)):
            lengths = list(shape)
        else:
            shape = torch_core.convert_to_tensor(shape, dtype="int64")
            lengths = [shape[i] for i in range(shape.numel())]

        # Dimensions whose start is a tensor → potentially dynamic.
        tensor_dims = []
        for dim, start in enumerate(starts):
            if isinstance(start, torch.Tensor):
                tensor_dims.append(dim)

        # No dynamic starts → fall back to the original implementation.
        if len(tensor_dims) == 0:
            return original_slice(inputs, starts, lengths)

        # Single dynamic dimension → use index_select to avoid unbacked
        # symbols in torch.export.
        if len(tensor_dims) == 1:
            dim = tensor_dims[0]
            start = starts[dim].reshape(())
            length = lengths[dim]

            # Build the selection indices dynamically.
            if isinstance(length, int):
                indices = torch.arange(
                    length, dtype=torch.int64, device=inputs.device
                )
            else:
                # length is a SymInt or tensor – try arange with it.
                indices = torch.arange(
                    length, dtype=torch.int64, device=inputs.device
                )
            indices = indices + start
            result = torch.index_select(inputs, dim, indices)

            # Apply static slicing for the remaining dimensions.
            for d, (s, l) in enumerate(zip(starts, lengths)):
                if d != dim and (s != 0 or l != result.shape[d]):
                    result = torch.narrow(result, d, s, l)
            return result

        # Multiple dynamic dimensions – fall back (will likely fail in export).
        return original_slice(inputs, starts, lengths)

    torch_core.slice_update = _patched_slice_update
    torch_core.slice = _patched_slice
    try:
        yield
    finally:
        torch_core.slice_update = original_slice_update
        torch_core.slice = original_slice
