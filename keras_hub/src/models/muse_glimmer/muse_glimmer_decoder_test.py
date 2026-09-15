import numpy as np
from keras import ops

from keras_hub.src.models.muse_glimmer.muse_glimmer_decoder import (
    MuseGlimmerTextDecoder,
)
from keras_hub.src.tests.test_case import TestCase


class MuseGlimmerTextDecoderTest(TestCase):
    def test_forward_full_attention_nope(self):
        layer = MuseGlimmerTextDecoder(
            intermediate_dim=16,
            num_query_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            use_rope=False,
            sliding_window_size=None,
        )
        x = ops.convert_to_tensor(np.random.randn(2, 5, 8).astype("float32"))
        padding_mask = ops.ones((2, 5), dtype="int32")
        output = layer(x, decoder_padding_mask=padding_mask)
        self.assertEqual(ops.shape(output), (2, 5, 8))

    def test_forward_sliding_window_rope(self):
        layer = MuseGlimmerTextDecoder(
            intermediate_dim=16,
            num_query_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            use_rope=True,
            sliding_window_size=2,
        )
        x = ops.convert_to_tensor(np.random.randn(2, 5, 8).astype("float32"))
        padding_mask = ops.ones((2, 5), dtype="int32")
        output = layer(x, decoder_padding_mask=padding_mask)
        self.assertEqual(ops.shape(output), (2, 5, 8))

    def test_get_config(self):
        layer = MuseGlimmerTextDecoder(
            intermediate_dim=16,
            num_query_heads=4,
            num_key_value_heads=2,
            head_dim=4,
        )
        config = layer.get_config()
        restored = MuseGlimmerTextDecoder.from_config(config)
        self.assertEqual(restored.intermediate_dim, 16)

    def test_use_bidirectional_attention_mask(self):
        layer = MuseGlimmerTextDecoder(
            intermediate_dim=16,
            num_query_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            use_bidirectional_attention=True,
            sliding_window_size=None,
        )
        x = ops.convert_to_tensor(np.random.randn(2, 5, 8).astype("float32"))
        padding_mask = ops.ones((2, 5), dtype="int32")
        mask = layer._compute_self_attention_mask(
            decoder_sequence=x,
            context_hidden_states=None,
            decoder_padding_mask=padding_mask,
            decoder_attention_mask=None,
            self_attention_cache=None,
            self_attention_cache_update_index=None,
        )
        # Bidirectional: every query position attends to every key
        # position (no causal triangle), unlike the default causal mask.
        self.assertEqual(ops.shape(mask), (2, 5, 5))
        self.assertTrue(bool(ops.all(mask)))

    def test_use_bidirectional_attention_with_context(self):
        layer = MuseGlimmerTextDecoder(
            intermediate_dim=16,
            num_query_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            use_bidirectional_attention=True,
            sliding_window_size=3,
        )
        x = ops.convert_to_tensor(np.random.randn(2, 4, 8).astype("float32"))
        context = ops.convert_to_tensor(
            np.random.randn(2, 3, 8).astype("float32")
        )
        padding_mask = ops.ones((2, 4), dtype="int32")
        output = layer(
            x,
            context_hidden_states=context,
            decoder_padding_mask=padding_mask,
        )
        self.assertEqual(ops.shape(output), (2, 4, 8))

    def test_bidirectional_attention_mask_with_persistent_cache(self):
        # DFlash context caching: keys span [cache buffer, fresh block].
        # Only cache slots at or before `anchor_pos` are valid (already
        # written); later slots are unwritten and must be masked out. A
        # sliding window further restricts by absolute-position distance.
        layer = MuseGlimmerTextDecoder(
            intermediate_dim=16,
            num_query_heads=1,
            num_key_value_heads=1,
            head_dim=4,
            use_bidirectional_attention=True,
            sliding_window_size=3,
        )
        max_length, q_len, anchor_pos = 6, 2, 2
        decoder_sequence = ops.zeros((1, q_len, 8))
        context_hidden_states = ops.zeros((1, 1, 8))
        self_attention_cache = ops.zeros((1, 2, max_length, 1, 4))
        mask = layer._compute_self_attention_mask(
            decoder_sequence=decoder_sequence,
            context_hidden_states=context_hidden_states,
            decoder_padding_mask=ops.ones((1, q_len), dtype="int32"),
            decoder_attention_mask=None,
            self_attention_cache=self_attention_cache,
            self_attention_cache_update_index=anchor_pos,
        )
        self.assertEqual(ops.shape(mask), (1, q_len, max_length + q_len))
        mask_np = ops.convert_to_numpy(mask)[0]

        window = 3
        for i in range(q_len):
            q_pos = anchor_pos + 1 + i
            expected = np.zeros((max_length + q_len,), dtype=bool)
            for key_pos in range(max_length + q_len):
                # Absolute key position: context slots are their own
                # index; block slots continue right after anchor_pos.
                abs_key_pos = (
                    key_pos
                    if key_pos < max_length
                    else anchor_pos + 1 + (key_pos - max_length)
                )
                is_valid_context = (
                    key_pos <= anchor_pos or key_pos >= max_length
                )
                in_window = abs(q_pos - abs_key_pos) <= (window - 1)
                expected[key_pos] = is_valid_context and in_window
            np.testing.assert_array_equal(mask_np[i], expected)

    def test_enable_qk_scale_and_gate_false(self):
        layer = MuseGlimmerTextDecoder(
            intermediate_dim=16,
            num_query_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            enable_qk_scale_and_gate=False,
        )
        x = ops.convert_to_tensor(np.random.randn(2, 5, 8).astype("float32"))
        padding_mask = ops.ones((2, 5), dtype="int32")
        output = layer(x, decoder_padding_mask=padding_mask)
        self.assertEqual(ops.shape(output), (2, 5, 8))
        self.assertFalse(hasattr(layer._self_attention_layer, "_gate_dense"))

    def test_use_sandwich_norm_false(self):
        layer = MuseGlimmerTextDecoder(
            intermediate_dim=16,
            num_query_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            use_sandwich_norm=False,
        )
        x = ops.convert_to_tensor(np.random.randn(2, 5, 8).astype("float32"))
        padding_mask = ops.ones((2, 5), dtype="int32")
        output = layer(x, decoder_padding_mask=padding_mask)
        self.assertEqual(ops.shape(output), (2, 5, 8))
        self.assertFalse(hasattr(layer, "_post_attention_layernorm"))
        self.assertFalse(hasattr(layer, "_post_feedforward_layernorm"))
