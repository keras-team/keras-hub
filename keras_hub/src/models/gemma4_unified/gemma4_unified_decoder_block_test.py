import keras
import numpy as np
from keras import ops

from keras_hub.src.models.gemma4_unified.gemma4_unified_decoder_block import (
    Gemma4UnifiedTextDecoderBlock,
)
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.keras_utils import running_on_gpu


class Gemma4UnifiedDecoderBlockTest(TestCase):
    def setUp(self):
        self.batch_size = 3
        self.total_sequence_length = 10
        self.hidden_dim = 512

        self.decoder_block = Gemma4UnifiedTextDecoderBlock(
            hidden_dim=self.hidden_dim,
            intermediate_dim=64,
            head_dim=64,
            num_query_heads=8,
            num_key_value_heads=8,
        )

        self.dummy_input = np.random.rand(
            self.batch_size, self.total_sequence_length, self.hidden_dim
        )

    def _build_causal_mask(self):
        mask = np.zeros(
            (
                self.batch_size,
                self.total_sequence_length,
                self.total_sequence_length,
            )
        )
        for i in range(self.total_sequence_length):
            mask[:, i, : i + 1] = True
        return mask

    def test_attention_mask_computation(self):
        attn_mask = self.decoder_block._compute_attention_mask(
            x=self.dummy_input,
            padding_mask=None,
            vision_mask=None,
            cache=None,
            cache_update_index=0,
        )
        expected_mask = self._build_causal_mask()
        self.assertAllEqual(
            expected_mask.astype(bool),
            ops.convert_to_numpy(attn_mask).astype(bool),
        )

    def test_attention_mask_with_vision_mask(self):
        block = Gemma4UnifiedTextDecoderBlock(
            hidden_dim=self.hidden_dim,
            intermediate_dim=64,
            head_dim=64,
            num_query_heads=8,
            num_key_value_heads=8,
            use_vision_bidirectional_attention=True,
        )
        vision_mask, expected_mask = self._vision_fixture()

        attn_mask = block._compute_attention_mask(
            x=self.dummy_input,
            padding_mask=None,
            vision_mask=vision_mask,
            cache=None,
            cache_update_index=0,
        )
        self.assertAllEqual(
            expected_mask.astype(bool),
            ops.convert_to_numpy(attn_mask).astype(bool),
        )

    def test_attention_mask_with_dual_masks(self):
        block = Gemma4UnifiedTextDecoderBlock(
            hidden_dim=self.hidden_dim,
            intermediate_dim=64,
            head_dim=64,
            num_query_heads=8,
            num_key_value_heads=8,
            use_vision_bidirectional_attention=True,
        )
        padding_mask = np.full(
            (self.batch_size, self.total_sequence_length), True
        )
        padding_mask[1, 6:] = False
        vision_mask, expected_mask = self._vision_fixture()

        attn_mask = block._compute_attention_mask(
            x=self.dummy_input,
            padding_mask=padding_mask,
            vision_mask=vision_mask,
            cache=None,
            cache_update_index=0,
        )

        # Padding on sample 1 masks out rows 6-9, image block included.
        for i in range(6, 10):
            expected_mask[1, i, 6:] = False

        self.assertAllEqual(
            expected_mask.astype(bool),
            ops.convert_to_numpy(attn_mask).astype(bool),
        )

    def _vision_fixture(self):
        vision_mask = np.full(
            (self.batch_size, self.total_sequence_length), False
        )
        vision_mask[0, 3:6] = True
        vision_mask[1, 2:5] = True
        vision_mask[1, 6:9] = True
        expected_mask = self._build_causal_mask()
        expected_mask[0, 3, 3:6] = True
        expected_mask[0, 4, 4:6] = True
        expected_mask[1, 2, 2:5] = True
        expected_mask[1, 3, 3:5] = True
        expected_mask[1, 4, 4:5] = True
        expected_mask[1, 6, 6:9] = True
        expected_mask[1, 7, 7:9] = True
        expected_mask[1, 8, 8:9] = True
        return vision_mask, expected_mask

    def _global_vision_block(self):
        # A global layer with a window far smaller than the sequence: the
        # window must not apply, and the image overlay must.
        return Gemma4UnifiedTextDecoderBlock(
            hidden_dim=self.hidden_dim,
            intermediate_dim=64,
            head_dim=64,
            num_query_heads=8,
            num_key_value_heads=8,
            use_vision_bidirectional_attention=True,
            use_sliding_window_attention=True,
            sliding_window_size=4,
            is_global_attention=True,
        )

    def test_global_layer_applies_image_overlay(self):
        # HF `Gemma4UnifiedModel.forward` applies the blockwise overlay to
        # full-attention layers as well as sliding ones.
        vision_mask, expected_mask = self._vision_fixture()
        attn_mask = self._global_vision_block()._compute_attention_mask(
            x=self.dummy_input,
            padding_mask=None,
            vision_mask=vision_mask,
            cache=None,
            cache_update_index=0,
        )
        self.assertAllEqual(
            expected_mask.astype(bool),
            ops.convert_to_numpy(attn_mask).astype(bool),
        )

    def test_prefill_with_cache_applies_image_overlay(self):
        block = self._global_vision_block()
        vision_mask, expected_mask = self._vision_fixture()
        length = self.total_sequence_length
        # The cache is only read via `ops.shape(cache)[2]`, so zeros of the
        # right rank suffice.
        cache = np.zeros((self.batch_size, 2, length, 8, 64), "float32")
        # Prefill: the query spans the whole cache, overlay applies.
        attn_mask = block._compute_attention_mask(
            x=self.dummy_input,
            padding_mask=None,
            vision_mask=vision_mask,
            cache=cache,
            cache_update_index=0,
        )
        self.assertAllEqual(
            expected_mask.astype(bool),
            ops.convert_to_numpy(attn_mask).astype(bool),
        )
        # Decode step mid-sequence with the `(B, 1)` slice `verify_next`
        # passes with k = 0: the row must stay causal. With the padded overlay
        # only column 0 could leak and it is already visible, so the `q = 3`
        # window below is the gate's pin.
        index = length // 2
        attn_mask = block._compute_attention_mask(
            x=self.dummy_input[:, :1],
            padding_mask=None,
            vision_mask=np.full((self.batch_size, 1), True),
            cache=cache,
            cache_update_index=index,
        )
        expected_row = np.zeros((self.batch_size, 1, length), bool)
        expected_row[:, 0, : index + 1] = True
        self.assertAllEqual(
            expected_row, ops.convert_to_numpy(attn_mask).astype(bool)
        )
        # `verify_next` window at `safe_start == 0` with k = 2: the query is
        # shorter than the cache, so the all-True image slice must not open
        # columns 1 and 2 for row 0. Green on the previous guard too; this
        # window exists to keep "gate always True" red.
        attn_mask = block._compute_attention_mask(
            x=self.dummy_input[:, :3],
            padding_mask=None,
            vision_mask=np.full((self.batch_size, 3), True),
            cache=cache,
            cache_update_index=0,
        )
        expected_window = np.zeros((self.batch_size, 3, length), bool)
        for i in range(3):
            expected_window[:, i, : i + 1] = True
        self.assertAllEqual(
            expected_window, ops.convert_to_numpy(attn_mask).astype(bool)
        )

    def test_prefill_traces_with_dynamic_shapes(self):
        # The prefill rule must not truth-test `ops.shape(...)`: traced with
        # unknown sequence and cache dimensions those are symbolic scalars.
        # Measured: `Layer.__call__` is not AutoGraph-converted, so a Python
        # `if ... or a == b` on those scalars raises even under a default
        # `tf.function` (the `generate` and `export` paths); calling the
        # private `_compute_attention_mask` directly is converted to
        # `tf.cond` and would hide the error, so the layer is called here.
        if keras.config.backend() != "tensorflow":
            self.skipTest("Uses `tf.function` with unknown dimensions.")
        import tensorflow as tf

        block = self._global_vision_block()
        vision_mask, _ = self._vision_fixture()
        length = self.total_sequence_length
        x = self.dummy_input.astype("float32")
        cache = np.zeros((self.batch_size, 2, length, 8, 64), "float32")
        eager_out, eager_cache = block(
            x, vision_mask=vision_mask, cache=cache, cache_update_index=0
        )

        @tf.function(
            input_signature=[
                tf.TensorSpec([None, None, self.hidden_dim], tf.float32),
                tf.TensorSpec([None, None], tf.bool),
                tf.TensorSpec([None, 2, None, 8, 64], tf.float32),
            ]
        )
        def prefill(x, vision_mask, cache):
            return block(
                x, vision_mask=vision_mask, cache=cache, cache_update_index=0
            )

        traced_out, traced_cache = prefill(x, vision_mask, cache)
        # Same weights, same inputs: the dynamic trace must reproduce the
        # eager prefill, whose mask the tests above pin to `expected_mask`.
        # A mask defect moves values by O(1); the traced graph and eager
        # mode pick different cuBLAS/TF32 kernels on GPU, measured max abs
        # diff 3.79e-3 on A100 TF-CUDA (22% of elements off by ~1e-3), so
        # 2e-2 keeps over 5x headroom there. CPU is within 1e-3.
        tol = 2e-2 if running_on_gpu() else 1e-3
        self.assertAllClose(traced_out, eager_out, atol=tol)
        self.assertAllClose(traced_cache, eager_cache, atol=tol)

    def test_global_layer_has_layer_scalar(self):
        """All text-decoder layers have layer_scalar."""
        text_block = Gemma4UnifiedTextDecoderBlock(
            hidden_dim=self.hidden_dim,
            intermediate_dim=64,
            head_dim=64,
            num_query_heads=8,
            num_key_value_heads=8,
            is_text_layer=True,
        )
        # Build by calling with dummy data.
        text_block(self.dummy_input)
        self.assertTrue(hasattr(text_block, "layer_scalar"))

    def test_int8_quantization_matches_float_output(self):
        # The FFW sub-block must call its `EinsumDense` layers so that
        # `quantize("int8")` applies the per-channel `kernel_scale`.
        # Multiplying by `.kernel` directly skips the scale: measured over 6
        # seeds on jax and tf, the int8 output then diverges from float by
        # 0.48-1.10 (max abs), versus <= 0.11 through the layer calls.
        keras.utils.set_random_seed(42)
        block = Gemma4UnifiedTextDecoderBlock(
            hidden_dim=32,
            intermediate_dim=32,
            head_dim=16,
            num_query_heads=2,
            num_key_value_heads=2,
        )
        model = keras.Sequential([block])
        x = np.random.RandomState(42).rand(2, 4, 32).astype("float32")
        float_out = ops.convert_to_numpy(model(x)[0])
        model.quantize("int8")
        int8_out = ops.convert_to_numpy(model(x)[0])
        self.assertAllClose(int8_out, float_out, atol=0.25, rtol=0.0)

    def test_moe_validation(self):
        with self.assertRaisesRegex(ValueError, "`num_experts` must be set"):
            Gemma4UnifiedTextDecoderBlock(
                hidden_dim=32,
                intermediate_dim=32,
                head_dim=16,
                num_query_heads=2,
                num_key_value_heads=2,
                enable_moe_block=True,
                expert_intermediate_dim=32,
                # num_experts is None
            )

        with self.assertRaisesRegex(
            ValueError, "`expert_intermediate_dim` must be set"
        ):
            Gemma4UnifiedTextDecoderBlock(
                hidden_dim=32,
                intermediate_dim=32,
                head_dim=16,
                num_query_heads=2,
                num_key_value_heads=2,
                enable_moe_block=True,
                num_experts=2,
                # expert_intermediate_dim is None
            )
