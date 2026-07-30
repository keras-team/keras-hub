"""PyTorch adapters for exporting KerasHub CausalLM models to LiteRT-LM."""

import contextlib
import inspect
import threading

import torch
from torch import nn

# Global lock serializing export-time mutations of PyTorch's default device.
# This keeps _cpu_default_device_scope thread-safe without changing semantics.
_DEFAULT_DEVICE_LOCK = threading.Lock()


@contextlib.contextmanager
def _cpu_default_device_scope():
    """Temporarily force PyTorch's default device to CPU.

    A module-level lock serializes this scope so concurrent exports (or
    exports running alongside GPU work) cannot observe a partially-applied
    default device.
    """
    with _DEFAULT_DEVICE_LOCK:
        original_device = torch.get_default_device()
        torch.set_default_device("cpu")
        try:
            yield
        finally:
            torch.set_default_device(original_device)


class KerasHubLiteRTAdapter(nn.Module):
    """Adapter that wraps a KerasHub CausalLM for LiteRT-LM export.

    The adapter exposes `forward_prefill` and `forward_decode` signatures
    compatible with `litert_torch.signature(...)`:

    Inputs:
        tokens:       int32 [batch, seq_len]
        input_pos:    int32 [seq_len]   (position indices)
        kv_cache_k_0, kv_cache_v_0, ...: per-layer KV caches

    Outputs (prefill):
        kv_cache_k_0, kv_cache_v_0, ...: updated per-layer KV caches
        (no logits -- LiteRT-LM extracts last-token logits via decode)

    Outputs (decode):
        logits:       float [batch, seq_len, vocab_size]
        kv_cache_k_0, kv_cache_v_0, ...: updated per-layer KV caches

    The adapter delegates to ``export_spec.stack_kv_cache``/
    ``unstack_kv_cache`` to convert between the flat per-layer k/v tensors
    and the cache shape ``model.call_with_cache()`` expects for this family.
    """

    def __init__(
        self,
        keras_model,
        num_layers,
        cache_length,
        export_spec,
    ):
        super().__init__()
        self.keras_model = keras_model
        self.num_layers = num_layers
        self.cache_length = cache_length
        self.export_spec = export_spec

        # Cache the call_with_cache signature so we don't re-inspect it on every
        # forward pass during export tracing.
        call_params = set(
            inspect.signature(keras_model.call_with_cache).parameters.keys()
        )
        self._call_with_cache_params = call_params

    def forward_prefill(self, tokens, input_pos, **kv_cache):
        """Prefill step: process the full prompt at the given cache position.

        LiteRT-LM requires prefill to return **only** KV cache tensors
        (no logits). The runtime extracts the last-token logits internally
        via a dedicated decode step.

        ``input_pos`` is a 1-D int32 tensor (e.g. ``[0, 1, 2, ...]`` for the
        first turn, or ``[N, N+1, ...]`` for subsequent turns). The first
        element is used as the cache-update index so that prefill appends to
        the existing cache instead of overwriting from position 0.
        """
        cache = self.export_spec.stack_kv_cache(kv_cache, self.num_layers)
        # The first element of input_pos is the start position.
        cache_update_index = input_pos[0]

        call_kwargs = self._build_call_with_cache_kwargs()
        return self._call_with_cache(
            tokens, cache, cache_update_index, call_kwargs, return_logits=False
        )

    def forward_decode(self, tokens, input_pos, **kv_cache):
        """Decode step: process a single token at ``input_pos``.

        ``input_pos`` is a scalar int32 tensor (e.g. ``[3]``). It is passed
        directly as the cache-update index so that the value remains a tensor
        inside the exported graph and is not baked in as a Python constant.
        """
        cache = self.export_spec.stack_kv_cache(kv_cache, self.num_layers)
        # Squeeze to a 0-D tensor so Keras cache operations receive a scalar.
        cache_update_index = input_pos.reshape(())
        call_kwargs = self._build_call_with_cache_kwargs()
        return self._call_with_cache(
            tokens, cache, cache_update_index, call_kwargs, return_logits=True
        )

    def _call_with_cache(
        self, tokens, cache, cache_update_index, call_kwargs, return_logits
    ):
        """Run ``keras_model.call_with_cache`` and return updated KV caches."""
        # Only Gemma3n forces an override here (a full-length padding mask;
        # see ``Gemma3nSpec.get_forced_call_with_cache_kwargs``); every other
        # family is a no-op.
        call_kwargs.update(
            self.export_spec.get_forced_call_with_cache_kwargs(
                tokens, self.cache_length
            )
        )
        logits, _, updated_cache = self.keras_model.call_with_cache(
            tokens,
            cache,
            cache_update_index,
            **call_kwargs,
        )
        # Keras cache-update ops (`slice_update`/`scatter_update`) are
        # purely functional, so `updated_cache` is already a fresh,
        # non-aliased tensor -- no `.clone()` needed.
        outputs = self.export_spec.unstack_kv_cache(
            updated_cache, self.num_layers
        )
        if return_logits:
            outputs["logits"] = logits
        return outputs

    def _build_call_with_cache_kwargs(self):
        """Build kwargs dict for ``call_with_cache`` based on its signature."""
        params = self._call_with_cache_params
        values = {
            "padding_mask": None,
            "cache_update_mask": None,
        }
        return {k: v for k, v in values.items() if k in params}
