"""CPU unit tests for the paged-attention bridge shape/argument handling.

These exercise `vllm_paged_attention` with a recording stand-in for the
injected kernel, so no TPU/vLLM/tpu-inference is required.
"""

from keras import ops

from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.vllm import context as vllm_context
from keras_hub.src.vllm.attention import vllm_paged_attention


class RecordingKernel:
    """Stand-in for the serving model's published paged-attention function.

    Records every call and echoes back `(new_kv_cache, outputs)` where
    `outputs` has the query's flat `(num_tokens, num_heads * head_dim)`
    shape — matching the real kernel's output convention — so tests can
    assert on the arguments the bridge passed.
    """

    def __init__(self):
        self.calls = []

    def __call__(
        self,
        kv_cache,
        q,
        k,
        v,
        scale,
        head_size,
        num_heads,
        num_kv_heads,
        sliding_window=None,
        soft_cap=None,
    ):
        self.calls.append(
            dict(
                kv_cache=kv_cache,
                q=q,
                k=k,
                v=v,
                scale=scale,
                head_size=head_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                sliding_window=sliding_window,
                soft_cap=soft_cap,
            )
        )
        return ("UPDATED", kv_cache), q


def activate_serving(kernel, kv_caches=None, positions=None):
    """Publishes a serving context wired to `kernel` for a test."""
    vllm_context.set_vllm_context(
        block_tables=None,
        slot_mapping=None,
        attention_metadata="META",
        paged_attention_func=kernel,
        mesh="MESH",
        positions=positions,
        kv_caches=kv_caches,
    )


class VllmPagedAttentionTest(TestCase):
    def tearDown(self):
        vllm_context.clear_vllm_context()
        super().tearDown()

    def test_returns_none_when_context_inactive(self):
        vllm_context.clear_vllm_context()
        z = ops.zeros((1, 2, 4, 8))
        self.assertIsNone(vllm_paged_attention(z, z, z, 0.125))

    def test_shape_conversion_and_argument_order(self):
        kernel = RecordingKernel()
        activate_serving(kernel, kv_caches=["CACHE0"])
        B, T, H, D = 1, 2, 4, 8
        q = ops.reshape(
            ops.arange(B * T * H * D, dtype="float32"), (B, T, H, D)
        )
        k = ops.ones((B, T, H, D))
        v = ops.ones((B, T, H, D))

        out = vllm_paged_attention(q, k, v, 0.125)

        # Output is restored to KerasHub's (B, T, H, D) layout.
        self.assertEqual(tuple(out.shape), (B, T, H, D))

        call = kernel.calls[0]
        # The published function's contract carries only what the layer
        # knows; engine-side args stay on the tpu-inference side.
        self.assertEqual(call["kv_cache"], "CACHE0")
        self.assertEqual(tuple(call["q"].shape), (B * T, H * D))
        self.assertEqual(tuple(call["k"].shape), (B * T, H * D))
        self.assertEqual(tuple(call["v"].shape), (B * T, H * D))
        self.assertEqual(call["scale"], 0.125)
        self.assertEqual(call["head_size"], D)
        self.assertEqual(call["num_heads"], H)
        self.assertEqual(call["num_kv_heads"], H)
        self.assertIsNone(call["sliding_window"])  # None = disabled
        self.assertIsNone(call["soft_cap"])  # None = disabled

    def test_caches_consumed_in_layer_order_and_updates_stored(self):
        kernel = RecordingKernel()
        activate_serving(kernel, kv_caches=["CACHE0", "CACHE1"])
        z = ops.zeros((1, 2, 4, 8))

        vllm_paged_attention(z, z, z, 0.125)
        vllm_paged_attention(z, z, z, 0.125)

        self.assertEqual(kernel.calls[0]["kv_cache"], "CACHE0")
        self.assertEqual(kernel.calls[1]["kv_cache"], "CACHE1")
        ctx = vllm_context.get_vllm_context()
        self.assertEqual(
            ctx.updated_kv_caches,
            [("UPDATED", "CACHE0"), ("UPDATED", "CACHE1")],
        )

        # A new forward step resets the layer counter.
        activate_serving(kernel, kv_caches=["FRESH"])
        vllm_paged_attention(z, z, z, 0.125)
        self.assertEqual(kernel.calls[2]["kv_cache"], "FRESH")

    def test_kv_heads_taken_from_key_for_gqa(self):
        kernel = RecordingKernel()
        activate_serving(kernel)
        q = ops.ones((1, 1, 8, 4))  # 8 query heads
        k = ops.ones((1, 1, 2, 4))  # 2 kv heads (grouped-query attention)
        v = ops.ones((1, 1, 2, 4))

        vllm_paged_attention(q, k, v, 0.5)

        call = kernel.calls[0]
        self.assertEqual(call["num_heads"], 8)  # from query
        self.assertEqual(call["num_kv_heads"], 2)  # from key

    def test_num_kv_heads_guard_matches_key(self):
        # When num_kv_heads matches the key's head count, it is a no-op.
        kernel = RecordingKernel()
        activate_serving(kernel)
        k = ops.ones((1, 1, 2, 4))
        out = vllm_paged_attention(
            ops.ones((1, 1, 8, 4)), k, k, 0.5, num_kv_heads=2
        )
        self.assertEqual(tuple(out.shape), (1, 1, 8, 4))

    def test_num_kv_heads_guard_rejects_expanded_kv(self):
        # A caller that expanded K/V (8 heads) but still reports 2 kv heads is
        # caught, rather than silently corrupting GQA in the kernel.
        activate_serving(RecordingKernel())
        expanded = ops.ones((1, 1, 8, 4))
        with self.assertRaises(ValueError):
            vllm_paged_attention(
                expanded, expanded, expanded, 0.5, num_kv_heads=2
            )

    def test_soft_cap_forwarded_only_when_set(self):
        kernel = RecordingKernel()
        activate_serving(kernel)
        q = ops.ones((1, 1, 2, 4))
        k = ops.ones((1, 1, 2, 4))
        v = ops.ones((1, 1, 2, 4))

        vllm_paged_attention(q, k, v, 0.5, soft_cap=50.0, sliding_window=64)

        call = kernel.calls[0]
        self.assertEqual(call["soft_cap"], 50.0)
        self.assertEqual(call["sliding_window"], 64)
