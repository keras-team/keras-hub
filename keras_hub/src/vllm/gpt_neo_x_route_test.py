"""CPU tests for GPT-NeoX's vLLM attention route.

Drives a real `GPTNeoXAttention` with a recording stand-in kernel, so no TPU
or vLLM install is needed.
"""

from keras import ops

from keras_hub.src.models.gpt_neo_x.gpt_neo_x_attention import GPTNeoXAttention
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.vllm import context as vllm_context
from keras_hub.src.vllm.attention_test import RecordingKernel
from keras_hub.src.vllm.attention_test import activate_serving


class GPTNeoXRouteTest(TestCase):
    def tearDown(self):
        vllm_context.clear_vllm_context()
        super().tearDown()

    def _build_layer(self, rotary_percentage=0.25):
        layer = GPTNeoXAttention(
            num_heads=2,
            hidden_dim=16,
            rotary_percentage=rotary_percentage,
        )
        inputs = ops.ones((3, 1, 16))
        layer.build(ops.shape(inputs))
        return layer, inputs

    def test_route_reports_head_counts_and_scale(self):
        layer, inputs = self._build_layer()

        kernel = RecordingKernel()
        activate_serving(
            kernel,
            kv_caches=["NC0"],
            positions=ops.convert_to_tensor([0, 1, 2]),
        )
        layer(inputs)

        call = kernel.calls[0]
        # GPT-NeoX has no grouped-query attention.
        self.assertEqual(call["num_heads"], 2)
        self.assertEqual(call["num_kv_heads"], 2)
        self.assertEqual(call["head_size"], layer.attn_head_size)
        self.assertAllClose(call["scale"], layer._inv_norm_factor)

    def test_route_rotates_only_the_rotary_slice(self):
        """Only the first `rotary_dim` features are rotated; the rest of
        each head passes through untouched."""
        layer, inputs = self._build_layer()

        kernel = RecordingKernel()
        positions = ops.convert_to_tensor([0, 1, 2])
        activate_serving(kernel, kv_caches=["NC0"], positions=positions)
        layer(inputs)

        query_key_value = layer._qkv_dense(inputs)
        query = query_key_value[..., : layer.attn_head_size]
        query_rot = layer.rotary_embedding_layer(
            query[..., : layer.rotary_dim],
            positions=ops.reshape(positions, (-1, 1)),
        )
        expected = ops.concatenate(
            (query_rot, query[..., layer.rotary_dim :]), axis=-1
        )
        self.assertAllClose(
            kernel.calls[0]["q"], ops.reshape(expected, (3, -1))
        )

    def test_route_returns_cache_unchanged(self):
        layer, inputs = self._build_layer()

        kernel = RecordingKernel()
        activate_serving(
            kernel,
            kv_caches=["NC0"],
            positions=ops.convert_to_tensor([0, 1, 2]),
        )
        output, cache = layer(inputs, cache="UNUSED")

        self.assertEqual(cache, "UNUSED")
        self.assertEqual(ops.shape(output), ops.shape(inputs))

    def test_off_path_byte_identical(self):
        layer, inputs = self._build_layer()
        # The dense path needs a mask: calling it without one trips a
        # backend-specific `Softmax` signature issue that predates this
        # route.
        mask = ops.ones((3, 1, 1), dtype="bool")

        before, _ = layer(inputs, attention_mask=mask)
        kernel = RecordingKernel()
        activate_serving(
            kernel,
            kv_caches=["NC0"],
            positions=ops.convert_to_tensor([0, 1, 2]),
        )
        vllm_context.clear_vllm_context()
        after, _ = layer(inputs, attention_mask=mask)

        self.assertAllClose(before, after)
        self.assertEqual(kernel.calls, [])
