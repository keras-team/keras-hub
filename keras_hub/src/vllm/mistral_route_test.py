"""CPU tests for Mistral's vLLM attention route.

Drives a real `CachedMistralAttention` with a recording stand-in kernel, so
no TPU or vLLM install is needed.
"""

from keras import ops

from keras_hub.src.models.mistral.mistral_attention import (
    CachedMistralAttention,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.vllm import context as vllm_context
from keras_hub.src.vllm.attention_test import RecordingKernel
from keras_hub.src.vllm.attention_test import activate_serving


class MistralRouteTest(TestCase):
    def tearDown(self):
        vllm_context.clear_vllm_context()
        super().tearDown()

    def _build_layer(self, sliding_window=512):
        layer = CachedMistralAttention(
            num_query_heads=4,
            num_key_value_heads=2,
            sliding_window=sliding_window,
        )
        inputs = ops.ones((3, 1, 8))
        layer.build(ops.shape(inputs))
        return layer, inputs

    def test_route_passes_unexpanded_kv_and_sliding_window(self):
        layer, inputs = self._build_layer()

        kernel = RecordingKernel()
        activate_serving(
            kernel,
            kv_caches=["MC0"],
            positions=ops.convert_to_tensor([0, 1, 2]),
        )
        layer(inputs)

        call = kernel.calls[0]
        # The kernel expands K/V for grouped-query attention itself, so the
        # route must hand it the unexpanded head count.
        self.assertEqual(call["num_kv_heads"], 2)
        self.assertEqual(call["num_heads"], 4)
        self.assertEqual(call["sliding_window"], 512)
        self.assertAllClose(call["scale"], layer._inv_norm_factor)

    def test_route_forwards_no_sliding_window_when_unset(self):
        layer, inputs = self._build_layer(sliding_window=None)

        kernel = RecordingKernel()
        activate_serving(
            kernel,
            kv_caches=["MC0"],
            positions=ops.convert_to_tensor([0, 1, 2]),
        )
        layer(inputs)

        self.assertIsNone(kernel.calls[0]["sliding_window"])

    def test_off_path_byte_identical(self):
        layer, inputs = self._build_layer()

        # No serving context: the dense path must be untouched.
        before = layer(inputs)
        kernel = RecordingKernel()
        activate_serving(
            kernel,
            kv_caches=["MC0"],
            positions=ops.convert_to_tensor([0, 1, 2]),
        )
        vllm_context.clear_vllm_context()
        after = layer(inputs)

        self.assertAllClose(before, after)
        self.assertEqual(kernel.calls, [])
