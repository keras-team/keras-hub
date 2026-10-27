"""Thread-local context for passing vLLM metadata to Keras layers.

The serving model (``KerasHubVllmModel`` in this package, registered
with vLLM by tpu-inference's plugin hook) publishes this
context around each forward step with ``vllm_context_scope``, which restores
the prior state on exit even when the forward raises, all on one thread.
KerasHub attention layers, the paged-attention bridge, and
``PositionEmbedding`` read it in-place, so serving metadata never has to
thread through layer signatures.

It is thread-local so concurrent requests on separate worker threads never see
each other's caches, positions, or kernel: each thread gets its own inactive
context until it calls ``set_vllm_context`` itself.
"""

import contextlib
import threading

# The single enumeration of the context's fields and their inactive values.
# `VLLMContext.__init__`, `clear_vllm_context`, and the scope's save/restore
# all derive from it, so adding a field means touching exactly one place.
_INACTIVE_STATE = {
    "paged_attention_func": None,
    "positions": None,
    "kv_caches": None,
    "updated_kv_caches": None,
    "layer_index": 0,
    "active": False,
}


def _reset(context):
    """Puts ``context`` into the inactive state."""
    for name, value in _INACTIVE_STATE.items():
        setattr(context, name, value)


class VLLMContext(threading.local):
    """Thread-local context for passing serving metadata to Keras layers.

    Attributes:
        paged_attention_func: The paged-attention closure published by the
            serving model; engine details are already bound inside it.
        positions: vLLM's per-token absolute position ids (for RoPE models).
        kv_caches: Per-layer paged KV caches for the current forward step.
            Attention layers consume these in call order via `layer_index`,
            so caches never need threading through layer signatures.
        updated_kv_caches: The per-layer caches returned by the kernel,
            collected by the serving model after the forward step.
        layer_index: Index of the next attention layer's cache; reset to 0
            by `set_vllm_context` at the start of every forward step.
        active: Boolean indicating if the context is currently active.
    """

    def __init__(self):
        """Initializes an empty inactive context."""
        super().__init__()
        _reset(self)


_vllm_context = VLLMContext()


def set_vllm_context(
    paged_attention_func=None,
    positions=None,
    kv_caches=None,
):
    """Sets the thread-local vLLM context parameters.

    Args:
        paged_attention_func: The paged-attention closure to dispatch to.
        positions: vLLM's per-token absolute position ids (used by RoPE models
            to apply rotary embeddings at the correct positions under paged /
            continuous-batched decode).
        kv_caches: Per-layer paged KV caches, in transformer-layer order.
    """
    _vllm_context.paged_attention_func = paged_attention_func
    _vllm_context.positions = positions
    _vllm_context.kv_caches = list(kv_caches) if kv_caches is not None else None
    _vllm_context.updated_kv_caches = (
        list(kv_caches) if kv_caches is not None else None
    )
    _vllm_context.layer_index = 0
    # Set last, so a failure partway through can never look half-active.
    _vllm_context.active = True


def clear_vllm_context():
    """Resets the thread-local vLLM context to its inactive state."""
    _reset(_vllm_context)


def get_vllm_context():
    """Retrieves the active thread-local vLLM context.

    Returns:
        The `VLLMContext` instance if active, otherwise `None`.
    """
    return _vllm_context if getattr(_vllm_context, "active", False) else None


@contextlib.contextmanager
def vllm_context_scope(
    paged_attention_func=None,
    positions=None,
    kv_caches=None,
):
    """Publishes the serving context for the enclosed forward step.

    Saves the prior state on entry and always restores it on exit,
    including when the forward raises — so a reused worker thread can never
    observe a stale context, and nested scopes hand back the outer scope's
    state instead of wiping it. Arguments match ``set_vllm_context``.
    """
    previous = {name: getattr(_vllm_context, name) for name in _INACTIVE_STATE}
    set_vllm_context(
        paged_attention_func=paged_attention_func,
        positions=positions,
        kv_caches=kv_caches,
    )
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(_vllm_context, name, value)
