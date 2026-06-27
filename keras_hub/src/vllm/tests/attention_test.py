"""CPU unit tests for the paged-attention bridge shape/argument handling.

These exercise `maybe_vllm_paged_attention` with a recording stand-in for the
injected kernel, so no TPU/vLLM/tpu-inference is required — only keras (any
backend) + numpy.
"""

from keras import ops

from keras_hub.src.vllm import context as vllm_context
from keras_hub.src.vllm.attention import maybe_vllm_paged_attention


class _RecordingKernel:
    """Stands in for tpu-inference's _jax_attn_func.

    Records the call and echoes back (new_kv_cache, outputs) where outputs has
    the same flat (num_tokens, num_heads*head_dim) shape as the query input,
    matching the real kernel's output convention.
    """

    def __init__(self):
        self.args = None
        self.kwargs = None

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        kv_cache, q = args[0], args[1]
        return kv_cache, q


def teardown_function(_):
    vllm_context.clear_vllm_context()


def test_returns_none_when_context_inactive():
    vllm_context.clear_vllm_context()
    z = ops.zeros((1, 2, 4, 8))
    assert maybe_vllm_paged_attention(z, z, z, None, 0.125) is None


def test_shape_conversion_and_argument_order():
    kernel = _RecordingKernel()
    cache = ops.zeros((3, 4))
    vllm_context.set_vllm_context(
        block_tables=None,
        slot_mapping=None,
        attention_metadata="META",
        paged_attention_func=kernel,
        mesh="MESH",
    )
    B, T, H, D = 1, 2, 4, 8
    q = ops.reshape(ops.arange(B * T * H * D, dtype="float32"), (B, T, H, D))
    k = ops.ones((B, T, H, D))
    v = ops.ones((B, T, H, D))

    out, new_cache = maybe_vllm_paged_attention(q, k, v, cache, scale=0.125)

    # Output is restored to KerasHub's (B, T, H, D) layout.
    assert tuple(out.shape) == (B, T, H, D)

    args, kwargs = kernel.args, kernel.kwargs
    # Order mirrors _jax_attn_func:
    # (kv_cache, q, k, v, sinks, attn_meta, mesh, scale, head_size,
    #  num_heads, num_kv_heads)
    assert args[0] is cache
    assert tuple(args[1].shape) == (B * T, H * D)  # flattened query
    assert tuple(args[2].shape) == (B * T, H * D)  # flattened key
    assert tuple(args[3].shape) == (B * T, H * D)  # flattened value
    assert args[4] is None  # sinks
    assert args[5] == "META"
    assert args[6] == "MESH"
    assert args[7] == 0.125  # scale
    assert args[8] == D  # head_size
    assert args[9] == H  # num_heads
    assert args[10] == H  # num_kv_heads
    assert kwargs.get("sliding_window") is None
    assert "soft_cap" not in kwargs  # omitted when not provided


def test_num_kv_heads_inferred_for_gqa():
    kernel = _RecordingKernel()
    vllm_context.set_vllm_context(None, None, "META", kernel, "MESH")
    q = ops.ones((1, 1, 8, 4))  # 8 query heads
    k = ops.ones((1, 1, 2, 4))  # 2 kv heads (grouped-query attention)
    v = ops.ones((1, 1, 2, 4))

    maybe_vllm_paged_attention(q, k, v, ops.zeros((1, 1)), scale=0.5)

    assert kernel.args[9] == 8  # num_heads from query
    assert kernel.args[10] == 2  # num_kv_heads from key


def test_soft_cap_forwarded_only_when_set():
    kernel = _RecordingKernel()
    vllm_context.set_vllm_context(None, None, "META", kernel, "MESH")
    q = ops.ones((1, 1, 2, 4))
    k = ops.ones((1, 1, 2, 4))
    v = ops.ones((1, 1, 2, 4))

    maybe_vllm_paged_attention(
        q, k, v, ops.zeros((1, 1)), scale=0.5, soft_cap=50.0, sliding_window=64
    )
    assert kernel.kwargs["soft_cap"] == 50.0
    assert kernel.kwargs["sliding_window"] == 64
