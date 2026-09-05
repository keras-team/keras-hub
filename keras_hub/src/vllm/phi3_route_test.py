"""CPU tests for Phi-3's vLLM attention route.

Drives a real `Phi3Attention` with a recording stand-in kernel, so no TPU or
vLLM install is needed.
"""

from keras import ops

from keras_hub.src.models.phi3.phi3_attention import Phi3Attention
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.vllm import context as vllm_context
from keras_hub.src.vllm.attention_test import RecordingKernel
from keras_hub.src.vllm.attention_test import activate_serving


class Phi3RouteTest(TestCase):
    def tearDown(self):
        vllm_context.clear_vllm_context()
        super().tearDown()

    def _build_layer(self, **kwargs):
        layer = Phi3Attention(
            num_query_heads=4,
            num_key_value_heads=2,
            **kwargs,
        )
        inputs = ops.ones((3, 1, 8))
        layer.build(ops.shape(inputs))
        return layer, inputs

    def _build_su_layer(self):
        return self._build_layer(
            rope_scaling_type="su",
            rope_scaling_short_factor=[1.0],
            rope_scaling_long_factor=[2.0],
            pretraining_sequence_length=8,
            max_sequence_length=64,
        )

    def test_route_passes_unexpanded_kv_heads(self):
        layer, inputs = self._build_layer()

        kernel = RecordingKernel()
        activate_serving(
            kernel,
            kv_caches=["PC0"],
            positions=ops.convert_to_tensor([0, 1, 2]),
        )
        layer(inputs)

        call = kernel.calls[0]
        # The kernel expands K/V for grouped-query attention itself.
        self.assertEqual(call["num_kv_heads"], 2)
        self.assertEqual(call["num_heads"], 4)
        self.assertIsNone(call["sliding_window"])
        self.assertAllClose(call["scale"], layer._inv_norm_factor)

    def test_route_applies_rope_at_serving_positions(self):
        layer, inputs = self._build_layer()

        kernel = RecordingKernel()
        positions = ops.convert_to_tensor([0, 1, 2])
        activate_serving(kernel, kv_caches=["PC0"], positions=positions)
        layer(inputs)

        expected_query = layer.rotary_embedding_layer(
            layer.query_dense(inputs),
            positions=ops.reshape(positions, (-1, 1)),
        )
        self.assertAllClose(
            kernel.calls[0]["q"], ops.reshape(expected_query, (3, -1))
        )

    def test_su_rope_matches_the_layers_own_embedding(self):
        """The route rebuilds su-scaled cos/sin because the rotary layer
        indexes a contiguous range. Within the pretraining length the two
        must agree exactly."""
        layer, inputs = self._build_layer(
            rope_scaling_type="su",
            rope_scaling_short_factor=[1.0],
            rope_scaling_long_factor=[2.0],
            pretraining_sequence_length=8,
            max_sequence_length=64,
        )

        kernel = RecordingKernel()
        activate_serving(
            kernel,
            kv_caches=["PC0"],
            positions=ops.convert_to_tensor([0, 1, 2]),
        )
        layer(inputs)

        # The same three tokens as one contiguous sequence, which is what
        # `Phi3SuScaledRotaryEmbedding` builds its own positions for.
        sequence = ops.ones((1, 3, 8))
        expected_query = layer.rotary_embedding_layer(
            layer.query_dense(sequence)
        )
        self.assertAllClose(
            kernel.calls[0]["q"], ops.reshape(expected_query, (3, -1))
        )

    def test_su_rope_uses_long_factor_far_into_the_sequence(self):
        """A single decoded token past the pretraining length must get
        long-context frequencies, which keying on the input's length would
        never do."""
        layer, inputs = self._build_su_layer()

        near = RecordingKernel()
        activate_serving(
            near,
            kv_caches=["PC0"],
            positions=ops.convert_to_tensor([0, 1, 2]),
        )
        layer(inputs)
        vllm_context.clear_vllm_context()

        far = RecordingKernel()
        activate_serving(
            far,
            kv_caches=["PC0"],
            positions=ops.convert_to_tensor([100, 101, 102]),
        )
        layer(inputs)

        self.assertNotAllClose(near.calls[0]["q"], far.calls[0]["q"])

    def test_off_path_byte_identical(self):
        layer, inputs = self._build_layer()

        before = layer(inputs)
        kernel = RecordingKernel()
        activate_serving(
            kernel,
            kv_caches=["PC0"],
            positions=ops.convert_to_tensor([0, 1, 2]),
        )
        vllm_context.clear_vllm_context()
        after = layer(inputs)

        self.assertAllClose(before, after)
        self.assertEqual(kernel.calls, [])
