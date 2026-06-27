"""Shared dispatch from KerasHub attention layers to vLLM's paged-attention.

KerasHub has several bespoke attention layers (the generic
``CachedMultiHeadAttention`` used by GPT-2 and friends, plus per-model layers
for Gemma, Llama, Mistral, ...). Rather than duplicate the (fragile) call into
``tpu_inference``'s native Pallas kernel in each one, every layer routes through
``maybe_vllm_paged_attention`` so the bridge lives in exactly one place.
"""

from typing import Any, Optional, Tuple

from keras import ops

from keras_hub.src.vllm.context import get_vllm_context


def maybe_vllm_paged_attention(
    query: Any,
    key: Any,
    value: Any,
    cache: Any,
    scale: float,
    sliding_window: Optional[int] = None,
    soft_cap: Optional[float] = None,
) -> Optional[Tuple[Any, Any]]:
    """Run vLLM's injected Pallas paged-attention kernel, if active.

    Returns ``(attention_output, new_kv_cache)`` when running under vLLM's TPU
    engine (i.e. a paged-attention function has been injected into the active
    ``vllm_context``), or ``None`` otherwise so the caller falls back to its
    standard Keras attention path.

    Args:
        query: Post-projection (and, where applicable, post-RoPE) query tensor.
        key: Post-projection/post-RoPE key tensor. Must NOT be GQA-expanded —
            the kernel handles grouped-query attention via ``num_kv_heads``,
            which is inferred from this tensor's head dimension.
        value: Post-projection value tensor (also not GQA-expanded).
        cache: This layer's paged KV cache (vLLM layout), or ``None``.
        scale: Softmax scale to apply inside the kernel (e.g. ``1/sqrt(head_dim)``).
            Pass the model's own convention; do not pre-scale ``query``.
        sliding_window: Optional sliding-window size for local attention.

    Returns:
        ``(attention_output, new_kv_cache)`` or ``None``. ``attention_output``
        is returned in KerasHub's ``(batch, seq, num_heads, head_dim)`` layout,
        ready for the caller's output projection.

    Note:
        KerasHub emits ``(batch, seq, num_heads, head_dim)`` tensors, whereas
        ``tpu_inference``'s ``_jax_attn_func`` expects flattened
        ``(num_tokens, num_heads * head_dim)`` inputs and returns the same flat
        layout. We flatten the leading (batch, seq) dims into a single token
        dim before the call and restore them afterwards. This conversion is a
        best-effort match to the kernel's contract and is the primary thing to
        confirm on real TPU hardware (see the Colab parity test).
    """
    ctx = get_vllm_context()
    if ctx is None or ctx.paged_attention_func is None:
        return None

    num_heads = query.shape[-2]
    head_dim = query.shape[-1]
    num_kv_heads = key.shape[-2]
    # Leading dims (e.g. (batch, seq)) to restore on the output.
    lead_shape = tuple(query.shape[:-2])

    # vLLM convention: a single flattened token dim, channels = heads * head_dim.
    query = ops.reshape(query, (-1, num_heads * head_dim))
    key = ops.reshape(key, (-1, num_kv_heads * head_dim))
    value = ops.reshape(value, (-1, num_kv_heads * head_dim))

    # `soft_cap` is only forwarded when set (Gemma 2/3); omitting it keeps this
    # call compatible with tpu-inference builds whose kernel dispatch predates
    # soft-cap plumbing.
    extra = {"sliding_window": sliding_window}
    if soft_cap is not None:
        extra["soft_cap"] = soft_cap

    new_kv_cache, attention_output = ctx.paged_attention_func(
        cache,
        query,
        key,
        value,
        None,  # sinks: KerasHub attention layers have no attention sinks
        ctx.attention_metadata,
        ctx.mesh,
        scale,
        head_dim,
        num_heads,
        num_kv_heads,
        **extra,
    )

    # Kernel returns (num_tokens, num_heads * head_dim); restore KerasHub layout.
    attention_output = ops.reshape(
        attention_output, (*lead_shape, num_heads, head_dim)
    )
    return attention_output, new_kv_cache
