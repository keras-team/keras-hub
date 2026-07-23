"""CPU tests for the per-model vLLM attention routes.

Each supported attention layer routes to the shared paged-attention bridge
only while a serving context is active, and runs its original dense path
otherwise. These build real layers and drive them with a recording
stand-in kernel — no TPU/vLLM needed.
"""

from keras import ops

from keras_hub.src.layers.modeling.cached_multi_head_attention import (
    CachedMultiHeadAttention,
)
from keras_hub.src.layers.modeling.position_embedding import PositionEmbedding
from keras_hub.src.models.gemma.gemma_attention import CachedGemmaAttention
from keras_hub.src.models.gemma3.gemma3_attention import CachedGemma3Attention
from keras_hub.src.models.gpt2.gpt2_backbone import GPT2Backbone
from keras_hub.src.models.llama.llama_attention import LlamaAttention
from keras_hub.src.models.qwen.qwen_attention import QwenAttention
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.vllm import context as vllm_context
from keras_hub.src.vllm.attention_test import RecordingKernel
from keras_hub.src.vllm.attention_test import activate_serving


class ModelRoutesTest(TestCase):
    def tearDown(self):
        vllm_context.clear_vllm_context()
        super().tearDown()

    def test_mha_rejects_attention_scores_while_serving(self):
        layer = CachedMultiHeadAttention(num_heads=2, key_dim=4)
        x = ops.ones((2, 3, 8))
        layer.build(ops.shape(x), ops.shape(x))
        activate_serving(RecordingKernel(), kv_caches=["C0"])
        with self.assertRaisesRegex(ValueError, "return_attention_scores"):
            layer(x, x, return_attention_scores=True)

    def test_gpt2_mha_routes_only_while_serving(self):
        layer = CachedMultiHeadAttention(num_heads=2, key_dim=4)
        x = ops.ones((3, 1, 8))
        baseline = layer(query=x, value=x)  # builds; off-path

        kernel = RecordingKernel()
        activate_serving(kernel, kv_caches=["C0"])
        out = layer(query=x, value=x)

        self.assertEqual(len(kernel.calls), 1)
        call = kernel.calls[0]
        self.assertEqual(call["kv_cache"], "C0")
        self.assertAlmostEqual(call["scale"], 4**-0.5)  # 1/sqrt(key_dim)
        self.assertEqual(tuple(out.shape), (3, 1, 8))

        # Off-path again: no new kernel call, output unchanged.
        vllm_context.clear_vllm_context()
        off = layer(query=x, value=x)
        self.assertEqual(len(kernel.calls), 1)
        self.assertAllClose(off, baseline)

    def test_llama_route_gqa_and_positions(self):
        layer = LlamaAttention(num_query_heads=4, num_key_value_heads=2)
        x = ops.ones((3, 1, 8))
        layer.build(ops.shape(x))

        kernel = RecordingKernel()
        activate_serving(
            kernel,
            kv_caches=["LC0"],
            positions=ops.convert_to_tensor([4, 7, 12]),
        )
        out = layer(x)

        call = kernel.calls[0]
        head_dim = 8 // 4
        self.assertEqual(call["kv_cache"], "LC0")
        self.assertEqual(call["num_heads"], 4)  # query heads
        self.assertEqual(call["num_kv_heads"], 2)  # unexpanded
        self.assertEqual(tuple(call["k"].shape), (3, 2 * head_dim))
        self.assertEqual(tuple(out.shape), (3, 1, 8))

        # Positions drive RoPE: a different position set changes the query.
        # Use ops.convert_to_numpy (not np.array) so it works on the torch
        # backend, whose tensors track grad and reject a bare numpy() call.
        q_before = ops.convert_to_numpy(call["q"])
        activate_serving(
            kernel, kv_caches=["LC0"], positions=ops.zeros((3,), "int32")
        )
        layer(x)
        self.assertNotAllClose(
            q_before, ops.convert_to_numpy(kernel.calls[1]["q"])
        )

    def test_qwen_route_forwards_sliding_window(self):
        layer = QwenAttention(
            num_query_heads=4,
            num_key_value_heads=2,
            use_sliding_window_attention=True,
            sliding_window_size=512,
        )
        x = ops.ones((3, 1, 8))
        layer.build(ops.shape(x))

        kernel = RecordingKernel()
        activate_serving(
            kernel,
            kv_caches=["QC0"],
            positions=ops.convert_to_tensor([0, 1, 2]),
        )
        layer(x)

        call = kernel.calls[0]
        self.assertEqual(call["num_kv_heads"], 2)  # unexpanded
        self.assertEqual(call["sliding_window"], 512)

    def test_gemma_1_2_route_scale_soft_cap_and_sliding(self):
        # One class (CachedGemmaAttention) backs both Gemma 1 and Gemma 2;
        # Gemma 2 just enables sliding window + soft cap.
        layer = CachedGemmaAttention(
            head_dim=4,
            num_query_heads=4,
            num_key_value_heads=2,
            logit_soft_cap=30.0,
            use_sliding_window_attention=True,
            sliding_window_size=512,
        )
        x = ops.ones((3, 1, 16))
        layer.build(ops.shape(x))

        kernel = RecordingKernel()
        activate_serving(
            kernel,
            kv_caches=["GC0"],
            positions=ops.convert_to_tensor([0, 1, 2]),
        )
        out = layer(x)

        call = kernel.calls[0]
        self.assertEqual(call["scale"], 1.0 / (4**0.5))  # 1/sqrt(head_dim)
        self.assertEqual(call["num_kv_heads"], 2)  # unexpanded
        self.assertEqual(call["soft_cap"], 30.0)
        self.assertEqual(call["sliding_window"], 512)
        self.assertEqual(tuple(out.shape), (3, 1, 16))

    def test_gemma3_route_scale_soft_cap_and_sliding(self):
        layer = CachedGemma3Attention(
            head_dim=4,
            num_query_heads=4,
            num_key_value_heads=2,
            use_sliding_window_attention=True,
            sliding_window_size=512,
        )
        x = ops.ones((3, 1, 16))
        layer.build(ops.shape(x))

        kernel = RecordingKernel()
        activate_serving(
            kernel,
            kv_caches=["GC0"],
            positions=ops.convert_to_tensor([0, 1, 2]),
        )
        out = layer(x)

        call = kernel.calls[0]
        self.assertEqual(call["scale"], 1.0 / (4**0.5))  # 1/sqrt(head_dim)
        self.assertEqual(call["num_kv_heads"], 2)  # unexpanded
        self.assertEqual(call["sliding_window"], 512)
        self.assertEqual(tuple(out.shape), (3, 1, 16))

    def test_llama_off_path_byte_identical(self):
        layer = LlamaAttention(num_query_heads=4, num_key_value_heads=2)
        x = ops.ones((2, 3, 8))
        vllm_context.clear_vllm_context()
        a = layer(x)
        b = layer(x)
        self.assertAllClose(a, b)

    def test_position_embedding_reads_serving_positions(self):
        # Learned positions must honor vLLM's per-token positions so the
        # serving model can delegate the whole forward to the backbone.
        layer = PositionEmbedding(sequence_length=16)
        x = ops.ones((3, 1, 8))  # (num_tokens, 1, hidden) serving layout
        layer.build(ops.shape(x))

        positions = ops.convert_to_tensor([5, 2, 9])
        activate_serving(RecordingKernel(), positions=positions)
        served = layer(x)  # positions=None -> read from context
        vllm_context.clear_vllm_context()

        # Equivalent to asking for those positions explicitly, and distinct
        # from the default contiguous 0..seq_len slice.
        explicit = layer(x, positions=ops.reshape(positions, (-1, 1)))
        default = layer(x)
        self.assertAllClose(served, explicit)
        self.assertNotAllClose(served, default)

    def test_gpt2_backbone_delegation_dispatches_every_layer(self):
        # The serving model delegates the whole forward to the backbone;
        # verify the serving context threads through so every attention layer
        # dispatches to the kernel exactly once, and learned positions route.
        backbone = GPT2Backbone(
            vocabulary_size=32,
            num_layers=3,
            num_heads=2,
            hidden_dim=8,
            intermediate_dim=16,
            max_sequence_length=16,
        )
        token_ids = ops.convert_to_tensor([[3], [7], [1], [9]], "int32")
        padding_mask = ops.ones_like(token_ids)

        kernel = RecordingKernel()
        activate_serving(
            kernel,
            kv_caches=[f"C{i}" for i in range(3)],
            positions=ops.convert_to_tensor([0, 1, 2, 3]),
        )
        out = backbone(
            {"token_ids": token_ids, "padding_mask": padding_mask},
            training=False,
        )
        ctx = vllm_context.get_vllm_context()
        layer_index = ctx.layer_index
        updated = list(ctx.updated_kv_caches)
        vllm_context.clear_vllm_context()

        # One dispatch per transformer layer, caches consumed in order.
        self.assertEqual(len(kernel.calls), 3)
        self.assertEqual(layer_index, 3)
        self.assertEqual(
            [c["kv_cache"] for c in kernel.calls], ["C0", "C1", "C2"]
        )
        self.assertEqual([c for _, c in updated], ["C0", "C1", "C2"])
        self.assertEqual(tuple(out.shape), (4, 1, 8))
