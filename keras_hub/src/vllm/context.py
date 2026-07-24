"""Thread-local context for passing vLLM metadata to Keras layers.

The serving model (``KerasHubForCausalLM`` in this package, registered
with vLLM by tpu-inference's plugin hook) publishes this
context around each forward step with ``vllm_context_scope``, which clears
it on exit even when the forward raises, all on one thread. KerasHub
attention layers, the paged-attention bridge, and ``PositionEmbedding`` read
it in-place, so serving metadata never has to thread through layer
signatures.

It is thread-local so concurrent requests on separate worker threads never see
each other's caches, positions, or kernel: each thread gets its own inactive
context until it calls ``set_vllm_context`` itself.
"""

import contextlib
import threading


class VLLMContext(threading.local):
    """Thread-local context for passing serving metadata to Keras layers.

    Attributes:
        block_tables: The block tables tensor for paged attention.
        slot_mapping: The slot mapping tensor for paged attention.
        attention_metadata: The full attention metadata object from vLLM.
        paged_attention_func: The compiled paged-attention kernel.
        mesh: The JAX device mesh required by the paged attention kernel.
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
        self.block_tables = None
        self.slot_mapping = None
        self.attention_metadata = None
        self.paged_attention_func = None
        self.mesh = None
        self.positions = None
        self.kv_caches = None
        self.updated_kv_caches = None
        self.layer_index = 0
        self.active = False


_vllm_context = VLLMContext()


def set_vllm_context(
    block_tables,
    slot_mapping,
    attention_metadata=None,
    paged_attention_func=None,
    mesh=None,
    positions=None,
    kv_caches=None,
):
    """Sets the thread-local vLLM context parameters.

    Args:
        block_tables: Array representing memory blocks for key/value caching.
        slot_mapping: Array mapping sequence tokens to cache slots.
        attention_metadata: Additional hardware/framework specific metadata.
        paged_attention_func: The function to use for paged attention.
        mesh: The JAX device mesh the paged attention kernel shards across.
        positions: vLLM's per-token absolute position ids (used by RoPE models
            to apply rotary embeddings at the correct positions under paged /
            continuous-batched decode).
        kv_caches: Per-layer paged KV caches, in transformer-layer order.
    """
    _vllm_context.block_tables = block_tables
    _vllm_context.slot_mapping = slot_mapping
    _vllm_context.attention_metadata = attention_metadata
    _vllm_context.paged_attention_func = paged_attention_func
    _vllm_context.mesh = mesh
    _vllm_context.positions = positions
    _vllm_context.kv_caches = list(kv_caches) if kv_caches is not None else None
    _vllm_context.updated_kv_caches = (
        list(kv_caches) if kv_caches is not None else None
    )
    _vllm_context.layer_index = 0
    _vllm_context.active = True


def clear_vllm_context():
    """Clears the thread-local vLLM context."""
    _vllm_context.block_tables = None
    _vllm_context.slot_mapping = None
    _vllm_context.attention_metadata = None
    _vllm_context.paged_attention_func = None
    _vllm_context.mesh = None
    _vllm_context.positions = None
    _vllm_context.kv_caches = None
    _vllm_context.updated_kv_caches = None
    _vllm_context.layer_index = 0
    _vllm_context.active = False


def get_vllm_context():
    """Retrieves the active thread-local vLLM context.

    Returns:
        The `VLLMContext` instance if active, otherwise `None`.
    """
    return _vllm_context if getattr(_vllm_context, "active", False) else None


@contextlib.contextmanager
def vllm_context_scope(
    block_tables,
    slot_mapping,
    attention_metadata=None,
    paged_attention_func=None,
    mesh=None,
    positions=None,
    kv_caches=None,
):
    """Publishes the serving context for the enclosed forward step.

    Sets the thread-local context on entry and always clears it on exit,
    including when the forward raises, so a reused worker thread can never
    observe a stale context. Arguments match ``set_vllm_context``.
    """
    set_vllm_context(
        block_tables=block_tables,
        slot_mapping=slot_mapping,
        attention_metadata=attention_metadata,
        paged_attention_func=paged_attention_func,
        mesh=mesh,
        positions=positions,
        kv_caches=kv_caches,
    )
    try:
        yield
    finally:
        clear_vllm_context()
