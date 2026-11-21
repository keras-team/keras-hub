from keras import ops

from keras_hub.src.vllm.context import get_vllm_context


def vllm_paged_attention(
    query,
    key,
    value,
    scale,
    num_kv_heads=None,
    sliding_window=None,
    soft_cap=None,
):
    """Run vLLM's injected Pallas paged-attention kernel, if active.

    Returns the attention output when running under vLLM's TPU engine (i.e. a
    paged-attention function has been injected into the active
    ``vllm_context``), or ``None`` otherwise so the caller falls back to its
    standard Keras attention path.

    This is the single place KerasHub attention layers talk to the kernel:
    each routed layer's ``call`` starts with a guarded check that delegates
    to its ``_compute_vllm_attention``, which projects Q/K/V (applying
    rotary embeddings at the serving context's per-token positions, so
    decode stays correct under continuous batching) and dispatches here.

    The layer's paged KV cache is taken from the serving context in
    layer-call order (``context.kv_caches[context.layer_index]``), and the
    kernel's updated cache is stored back into
    ``context.updated_kv_caches`` for the serving model to collect.

    Args:
        query: Post-projection (and, where applicable, post-RoPE) query
            tensor in KerasHub's ``(batch, seq, num_heads, head_dim)`` layout.
        key: Post-projection/post-RoPE key tensor.
        value: Post-projection value tensor.
        scale: Softmax scale applied inside the kernel (1/sqrt(head_dim)).
            Pass the model's own convention; do not pre-scale ``query``.
        num_kv_heads: Optional consistency check on grouped-query attention.
            The kernel derives the KV head count from ``key`` (whose heads
            must be unexpanded), and this raises if the two disagree, catching
            a caller that expanded K/V before dispatching. When ``None``, no
            check is done.
        sliding_window: Optional sliding-window size for local attention.
        soft_cap: Optional attention-logit soft cap (e.g. Gemma 2). None
            disables capping in the kernel.

    Returns:
        The attention output in KerasHub's ``(batch, seq, num_heads,
        head_dim)`` layout, ready for the caller's output projection — or
        ``None`` when not serving.
    """
    ctx = get_vllm_context()
    if ctx is None or ctx.paged_attention_func is None:
        return None

    num_heads = query.shape[-2]
    head_dim = query.shape[-1]
    # Callers pass K/V with their own (unexpanded) head count; the kernel
    # handles grouped-query attention from `kv_heads`. `num_kv_heads`, when
    # given, is only a guard: it must match, catching a caller that expanded
    # K/V before dispatching (which would silently break GQA in the kernel).
    kv_heads = key.shape[-2]
    if num_kv_heads is not None and kv_heads != num_kv_heads:
        raise ValueError(
            "vllm_paged_attention expects unexpanded key/value tensors, but "
            f"got {kv_heads} kv head(s) while the layer reports "
            f"num_kv_heads={num_kv_heads}. Pass K/V before any grouped-query "
            "head expansion; the kernel expands internally."
        )
    # Leading dims (e.g. (batch, seq)) to restore on the output. Static by
    # construction: this code only runs on the JAX serving path, where
    # shapes are concrete ints even under jit tracing.
    lead_shape = tuple(query.shape[:-2])

    # This layer's paged cache, taken in layer-call order. A mismatch between
    # the number of attention layers and the caches vLLM allocated would
    # otherwise surface as a cryptic IndexError, so check explicitly.
    cache = None
    cache_index = None
    if ctx.kv_caches is not None:
        cache_index = ctx.layer_index
        if cache_index >= len(ctx.kv_caches):
            raise ValueError(
                f"vLLM paged attention was invoked {cache_index + 1} times "
                f"but only {len(ctx.kv_caches)} per-layer caches were "
                "provided. The model has more attention layers than vLLM "
                "allocated caches for."
            )
        ctx.layer_index = cache_index + 1
        cache = ctx.kv_caches[cache_index]

    # vLLM convention: one flattened token dim, channels = heads*head_dim.
    query = ops.reshape(query, (-1, num_heads * head_dim))
    key = ops.reshape(key, (-1, kv_heads * head_dim))
    value = ops.reshape(value, (-1, kv_heads * head_dim))

    # `paged_attention_func` comes from the serving model
    # (keras_hub_vllm_wrapper.py).
    # It takes only what an attention layer knows; engine details (the
    # attention metadata, the mesh) are already bound inside it. It returns
    # `(new_kv_cache, output)`.
    new_kv_cache, attention_output = ctx.paged_attention_func(
        cache,
        query,
        key,
        value,
        scale,
        head_dim,
        num_heads,
        kv_heads,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
    )

    if cache_index is not None:
        ctx.updated_kv_caches[cache_index] = new_kv_cache

    # Kernel returns (num_tokens, heads*head_dim); restore KerasHub layout.
    attention_output = ops.reshape(
        attention_output, (*lead_shape, num_heads, head_dim)
    )
    return attention_output
