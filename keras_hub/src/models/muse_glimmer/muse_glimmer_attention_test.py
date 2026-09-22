import numpy as np
from keras import ops

from keras_hub.src.models.muse_glimmer.muse_glimmer_attention import (
    MuseGlimmerTextAttention,
)
from keras_hub.src.tests.test_case import TestCase


class MuseGlimmerTextAttentionTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "num_query_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 4,
        }

    def test_call_shape(self):
        layer = MuseGlimmerTextAttention(
            **dict(
                self.init_kwargs,
                num_query_heads=4,
                qk_scale_factor=3.87,
                use_rope=True,
                sliding_window_size=None,
            )
        )
        self.run_serialization_test(layer)
        x = ops.convert_to_tensor(np.random.randn(2, 5, 8).astype("float32"))
        mask = ops.ones((2, 5, 5), dtype="bool")
        output = layer(x, attention_mask=mask)
        self.assertEqual(ops.shape(output), (2, 5, 8))
        self.assertGreater(len(layer.trainable_weights), 0)

    def test_nope_layer_ignores_position(self):
        layer = MuseGlimmerTextAttention(
            **dict(
                self.init_kwargs,
                num_key_value_heads=1,
                use_rope=False,
            )
        )
        self.run_serialization_test(layer)
        x = ops.convert_to_tensor(np.random.randn(1, 4, 8).astype("float32"))
        mask = ops.ones((1, 4, 4), dtype="bool")
        output = layer(x, attention_mask=mask)
        self.assertEqual(ops.shape(output), (1, 4, 8))
        self.assertFalse(hasattr(layer, "rotary_embedding_layer"))

    def test_sliding_window(self):
        layer = MuseGlimmerTextAttention(
            **dict(self.init_kwargs, use_rope=True, sliding_window_size=2)
        )
        self.run_serialization_test(layer)
        x = ops.convert_to_tensor(np.random.randn(1, 6, 8).astype("float32"))
        mask = ops.ones((1, 6, 6), dtype="bool")
        output = layer(x, attention_mask=mask)
        self.assertEqual(ops.shape(output), (1, 6, 8))

    def test_context_hidden_states_mismatched_length(self):
        # Assistant/drafter configuration: Q comes only from the block
        # window (q_len=5); K/V see context+block concatenated
        # (kv_len=3+5=8). RoPE must apply the trailing `q_len` positions
        # to Q and the full range to K without raising a shape error.
        layer = MuseGlimmerTextAttention(
            **dict(self.init_kwargs, use_rope=True, sliding_window_size=None)
        )
        hidden_states = ops.convert_to_tensor(
            np.random.randn(1, 5, 8).astype("float32")
        )
        context_hidden_states = ops.convert_to_tensor(
            np.random.randn(1, 3, 8).astype("float32")
        )
        mask = ops.ones((1, 5, 8), dtype="bool")
        output = layer(
            hidden_states,
            context_hidden_states=context_hidden_states,
            attention_mask=mask,
        )
        self.assertEqual(ops.shape(output), (1, 5, 8))

    def test_sliding_window_mask_offset_with_context(self):
        # Query occupies the trailing `query_len` rows of the combined
        # `context_len + query_len` key range. The sliding-window mask must
        # be sliced with that same offset (mirroring the RoPE offset in
        # `call()`), not from row 0, or every query row gets checked
        # against the wrong window of key columns.
        layer = MuseGlimmerTextAttention(
            **dict(
                self.init_kwargs,
                num_query_heads=1,
                num_key_value_heads=1,
                use_rope=False,
                sliding_window_size=2,
            )
        )
        query_len, context_len = 3, 4
        key_len = query_len + context_len
        attention_mask = ops.ones((1, query_len, key_len), dtype="bool")
        mask = layer._mask_sliding_window(
            attention_mask, cache_update_index=0, context_len=context_len
        )
        mask_np = ops.convert_to_numpy(mask)[0]
        # Row i of the mask corresponds to query position `context_len + i`
        # in the combined range, so it must allow keys in
        # `[context_len + i - (window - 1), context_len + i + (window - 1)]`.
        window = 2
        for i in range(query_len):
            q_pos = context_len + i
            expected = np.zeros((key_len,), dtype=bool)
            lo = max(0, q_pos - (window - 1))
            hi = min(key_len - 1, q_pos + (window - 1))
            expected[lo : hi + 1] = True
            np.testing.assert_array_equal(mask_np[i], expected)

    def test_context_cache_persists_across_cycles(self):
        # DFlash context caching: `cache` should only ever be written at
        # the context position (`cache_update_index`), never at the fresh
        # block's own positions — and a later cycle's context write must
        # not disturb an earlier cycle's already-cached slot.
        layer = MuseGlimmerTextAttention(
            **dict(
                self.init_kwargs,
                num_query_heads=1,
                num_key_value_heads=1,
                use_rope=False,
                sliding_window_size=None,
            )
        )
        max_length, block_size = 5, 2
        cache = ops.zeros((1, 2, max_length, 1, 4))
        mask = ops.ones((1, block_size, max_length + block_size), dtype="bool")

        context_0 = ops.convert_to_tensor(
            np.random.randn(1, 1, 8).astype("float32")
        )
        block_0 = ops.convert_to_tensor(
            np.random.randn(1, block_size, 8).astype("float32")
        )
        output_0, cache = layer(
            block_0,
            context_hidden_states=context_0,
            attention_mask=mask,
            cache=cache,
            cache_update_index=0,
        )
        self.assertEqual(ops.shape(output_0), (1, block_size, 8))
        key_cache_0 = ops.convert_to_numpy(cache[:, 0, 0, ...])

        # A second cycle writes a new context position further along;
        # the first cycle's cached slot must be unchanged.
        context_1 = ops.convert_to_tensor(
            np.random.randn(1, 1, 8).astype("float32")
        )
        block_1 = ops.convert_to_tensor(
            np.random.randn(1, block_size, 8).astype("float32")
        )
        output_1, cache = layer(
            block_1,
            context_hidden_states=context_1,
            attention_mask=mask,
            cache=cache,
            cache_update_index=1,
        )
        self.assertEqual(ops.shape(output_1), (1, block_size, 8))
        key_cache_0_after = ops.convert_to_numpy(cache[:, 0, 0, ...])
        self.assertAllClose(key_cache_0, key_cache_0_after)
        # The newly written slot must not still be all-zero (i.e. it was
        # actually written, not skipped).
        key_cache_1 = ops.convert_to_numpy(cache[:, 0, 1, ...])
        self.assertFalse(np.allclose(key_cache_1, np.zeros_like(key_cache_1)))

    def test_enable_qk_scale_and_gate_false(self):
        layer = MuseGlimmerTextAttention(
            **dict(
                self.init_kwargs,
                qk_scale_factor=3.87,
                use_rope=True,
                sliding_window_size=None,
                enable_qk_scale_and_gate=False,
            )
        )
        self.run_serialization_test(layer)
        x = ops.convert_to_tensor(np.random.randn(1, 5, 8).astype("float32"))
        mask = ops.ones((1, 5, 5), dtype="bool")
        output = layer(x, attention_mask=mask)
        self.assertEqual(ops.shape(output), (1, 5, 8))
        self.assertFalse(hasattr(layer, "_gate_dense"))
