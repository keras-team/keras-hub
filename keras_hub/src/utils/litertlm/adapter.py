"""PyTorch adapter modules for exporting KerasHub CausalLM models to LiteRT-LM."""

import contextlib

import torch
from torch import nn

from keras.src.backend.torch import core as torch_core


class KerasHubLiteRTAdapter(nn.Module):
    """Adapter that wraps a KerasHub CausalLM for LiteRT-LM export.

    The adapter exposes `forward_prefill` and `forward_decode` signatures
    compatible with `litert_torch.signature(...)`:

        inputs:
            tokens:       int32 [batch, seq_len]
            input_pos:    int32 [seq_len]   (position indices)
            mask:         optional float mask (ignored by most models)
            kv_cache_k_0, kv_cache_v_0, ...: per-layer KV caches

        outputs:
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
        """Prefill step – processes the full prompt starting at cache pos 0."""
        cache = self._stack_kv_cache(kv_cache)
        # Prefill always starts writing at position 0.
        logits, _, updated_cache = self.keras_model.call_with_cache(
            tokens, cache, 0
        )
        outputs = self._unstack_kv_cache(updated_cache)
        outputs["logits"] = logits
        return outputs

    def forward_decode(self, tokens, input_pos, mask=None, **kv_cache):
        """Decode step – processes a single token at *input_pos*.

        ``input_pos`` is a scalar int32 tensor (e.g. ``[3]``).  It is passed
        directly as the cache-update index so that the value remains a tensor
        inside the exported graph and is not baked in as a Python constant.
        """
        cache = self._stack_kv_cache(kv_cache)
        # input_pos is a 0-D or 1-D tensor; pass it through as the update index.
        cache_update_index = input_pos
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

    This patch detects the single dynamic dimension in a cache-update call
    (all other starts are constant zeros) and replaces the implementation
    with ``torch.index_copy_``, which accepts scalar tensors and is fully
    traceable.  Constant-index cases fall back to the original logic.
    """
    original = torch_core.slice_update

    def _patched(inputs, start_indices, updates):
        inputs = torch_core.convert_to_tensor(inputs)
        updates = torch_core.convert_to_tensor(updates)

        # Normalise start_indices to a list of scalar tensors.
        if isinstance(start_indices, (list, tuple)):
            starts = []
            for s in start_indices:
                if isinstance(s, torch.Tensor):
                    starts.append(s)
                else:
                    starts.append(
                        torch.tensor(s, dtype=torch.int64, device=inputs.device)
                    )
            start_indices = starts
        else:
            start_indices = torch_core.convert_to_tensor(
                start_indices, dtype="int64"
            )
            start_indices = [
                start_indices[i] for i in range(start_indices.numel())
            ]

        # Find dimensions whose start index is *dynamic* (i.e. not a
        # constant zero baked into the graph).
        non_const_dims = []
        for dim, start in enumerate(start_indices):
            if isinstance(start, torch.Tensor) and start.numel() == 1:
                start = start.reshape(())
                if start.dim() == 0:
                    try:
                        val = int(start.item())
                        if val != 0:
                            non_const_dims.append(dim)
                    except Exception:
                        # .item() failed → dynamic tensor input.
                        non_const_dims.append(dim)

        # If there is exactly one dynamic dimension and the update length
        # along that dimension is 1 (the decode case), use index_copy_.
        if len(non_const_dims) == 1:
            dim = non_const_dims[0]
            update_len = updates.shape[dim]
            if update_len == 1:
                start = start_indices[dim].reshape(())
                index = start.unsqueeze(0)
                outputs = torch.clone(inputs)
                outputs.index_copy_(dim, index, updates)
                return outputs

        # Fallback – all starts are constants (e.g. prefill with start=0).
        return original(inputs, [int(s.item()) for s in start_indices], updates)

    torch_core.slice_update = _patched
    try:
        yield
    finally:
        torch_core.slice_update = original
