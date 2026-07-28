"""CPU tests for SmolLM3's vLLM attention route.

Drives a real `SmolLM3Attention` with a recording stand-in kernel, so no TPU
or vLLM install is needed.
"""

from keras import ops

from keras_hub.src.models.smollm3.smollm3_layers import SmolLM3Attention
from keras_hub.src.models.smollm3.smollm3_utils import apply_rotary_pos_emb
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.vllm import context as vllm_context
from keras_hub.src.vllm.attention_test import RecordingKernel
from keras_hub.src.vllm.attention_test import activate_serving


class SmolLM3RouteTest(TestCase):
    def tearDown(self):
        vllm_context.clear_vllm_context()
        super().tearDown()

    def _build_layer(self, rope_layer_enabled_list=(True,)):
        layer = SmolLM3Attention(
            hidden_size=16,
            num_attention_heads=4,
            num_key_value_heads=2,
            attention_bias=False,
            attention_dropout=0.0,
            rope_layer_enabled_list=list(rope_layer_enabled_list),
            layer_types=["attention"],
            layer_idx=0,
        )
        inputs = ops.ones((3, 1, 16))
        # SmolLM3Attention.build takes a list of input shapes.
        layer.build([ops.shape(inputs)])
        return layer, inputs

    def test_route_passes_unexpanded_kv_heads_and_scale(self):
        layer, inputs = self._build_layer()

        kernel = RecordingKernel()
        activate_serving(
            kernel,
            kv_caches=["SC0"],
            positions=ops.convert_to_tensor([0, 1, 2]),
        )
        layer(inputs)

        call = kernel.calls[0]
        # The kernel expands K/V for grouped-query attention itself.
        self.assertEqual(call["num_kv_heads"], 2)
        self.assertEqual(call["num_heads"], 4)
        self.assertAllClose(call["scale"], layer.scaling)

    def test_route_rope_matches_the_layers_own_embedding(self):
        """The route rebuilds cos/sin from `inv_freq` because the rotary
        layer only walks a contiguous range. For positions 0..2 the two
        must agree exactly."""
        layer, inputs = self._build_layer()

        kernel = RecordingKernel()
        activate_serving(
            kernel,
            kv_caches=["SC0"],
            positions=ops.convert_to_tensor([0, 1, 2]),
        )
        layer(inputs)

        # Same three tokens as one contiguous sequence, which is the shape
        # `SmolLM3RotaryEmbedding` builds its own positions for.
        sequence = ops.ones((1, 3, 16))
        query = ops.reshape(
            layer.q_proj(sequence),
            (1, 3, layer.num_attention_heads, layer.head_dim),
        )
        key = ops.reshape(
            layer.k_proj(sequence),
            (1, 3, layer.num_key_value_heads, layer.head_dim),
        )
        cos, sin = layer.rotary_embedding(query)
        expected_query, _ = apply_rotary_pos_emb(
            query, key, cos, sin, expansion_axis=2
        )

        self.assertAllClose(
            kernel.calls[0]["q"], ops.reshape(expected_query, (3, -1))
        )

    def test_nope_layer_skips_rope(self):
        """SmolLM3 disables RoPE on some layers; the route must too."""
        layer, inputs = self._build_layer(rope_layer_enabled_list=(False,))

        kernel = RecordingKernel()
        activate_serving(
            kernel,
            kv_caches=["SC0"],
            positions=ops.convert_to_tensor([0, 1, 2]),
        )
        layer(inputs)

        # Same input rows, no rotation applied: every token matches.
        rotated = kernel.calls[0]["q"]
        self.assertAllClose(rotated[0], rotated[2])

    def test_off_path_byte_identical(self):
        layer, inputs = self._build_layer()

        before = layer(inputs)
        kernel = RecordingKernel()
        activate_serving(
            kernel,
            kv_caches=["SC0"],
            positions=ops.convert_to_tensor([0, 1, 2]),
        )
        vllm_context.clear_vllm_context()
        after = layer(inputs)

        self.assertAllClose(before, after)
        self.assertEqual(kernel.calls, [])
