import numpy as np
from keras import ops

from keras_hub.src.models.diffusion_gemma.diffusion_gemma_transformer_layer import (  # noqa: E501
    DiffusionGemmaTransformerLayer,
)
from keras_hub.src.tests.test_case import TestCase


class DiffusionGemmaTransformerLayerTest(TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 8
        # hidden_dim must equal head_dim * num_query_heads
        self.hidden_dim = 8
        self.head_dim = 4
        self.num_query_heads = 2
        self.num_key_value_heads = 2

        self.layer = DiffusionGemmaTransformerLayer(
            hidden_dim=self.hidden_dim,
            intermediate_dim=16,
            head_dim=self.head_dim,
            num_query_heads=self.num_query_heads,
            num_key_value_heads=self.num_key_value_heads,
        )

        self.dummy_input = np.random.randn(
            self.batch_size, self.seq_len, self.hidden_dim
        ).astype("float32")

    def test_output_shape(self):
        x_out, cache_out = self.layer(self.dummy_input)
        self.assertEqual(
            x_out.shape,
            (self.batch_size, self.seq_len, self.hidden_dim),
        )
        self.assertEqual(cache_out.shape[0], self.batch_size)
        self.assertEqual(cache_out.shape[2], self.seq_len)

    def test_text_layer_has_both_scalars_after_build(self):
        self.layer(self.dummy_input)
        self.assertTrue(hasattr(self.layer, "layer_scalar"))
        self.assertTrue(hasattr(self.layer, "encoder_layer_scalar"))

    def test_scalars_initialise_to_one(self):
        self.layer(self.dummy_input)
        self.assertAlmostEqual(
            float(ops.convert_to_numpy(self.layer.layer_scalar)), 1.0
        )
        self.assertAlmostEqual(
            float(ops.convert_to_numpy(self.layer.encoder_layer_scalar)), 1.0
        )

    def test_is_encoder_selects_correct_scalar(self):
        x_t = ops.convert_to_tensor(self.dummy_input)
        self.layer(x_t)

        self.layer.layer_scalar.assign(2.0)
        self.layer.encoder_layer_scalar.assign(3.0)

        out_decoder, _ = self.layer(x_t, is_encoder=False)
        out_encoder, _ = self.layer(x_t, is_encoder=True)

        self.assertNotAllClose(
            ops.convert_to_numpy(out_decoder),
            ops.convert_to_numpy(out_encoder),
        )

    def test_encoder_and_decoder_scalars_are_independent(self):
        x = ops.ones((1, 4, self.hidden_dim), dtype="float32")
        self.layer(x)

        self.layer.encoder_layer_scalar.assign(2.0)
        self.layer.layer_scalar.assign(0.5)
        out_enc, _ = self.layer(x, is_encoder=True)
        out_dec, _ = self.layer(x, is_encoder=False)
        self.assertNotAllClose(
            ops.convert_to_numpy(ops.stop_gradient(out_enc)),
            ops.convert_to_numpy(ops.stop_gradient(out_dec)),
        )

        # Symmetry: equal scalars → matching outputs.
        self.layer.encoder_layer_scalar.assign(0.5)
        out_enc_equal, _ = self.layer(x, is_encoder=True)
        self.assertAllClose(
            ops.convert_to_numpy(ops.stop_gradient(out_enc_equal)),
            ops.convert_to_numpy(ops.stop_gradient(out_dec)),
            atol=1e-5,
        )

    def test_canvas_bidirectional_mask_shape(self):
        self.layer(self.dummy_input)
        output_length = 4
        input_length = 8
        cache_update_index = 4

        canvas_mask = np.array(
            [[True, True, False, False], [False, True, True, False]], dtype=bool
        )
        mask = self.layer._compute_canvas_bidirectional_attention_mask(
            ops.convert_to_tensor(canvas_mask),
            cache_update_index=cache_update_index,
            output_length=output_length,
            input_length=input_length,
        )
        self.assertEqual(
            ops.convert_to_numpy(mask).shape,
            (2, output_length, input_length),
        )

    def test_canvas_mask_allows_canvas_to_canvas_attention(self):
        """Canvas query positions must attend to all canvas key positions."""
        self.layer(self.dummy_input)
        output_length = 4
        input_length = 10
        cache_update_index = 4  # canvas keys occupy positions 4..7

        canvas_mask = np.array(
            [[True, True, False, False], [False, False, True, True]], dtype=bool
        )
        mask_np = ops.convert_to_numpy(
            self.layer._compute_canvas_bidirectional_attention_mask(
                ops.convert_to_tensor(canvas_mask),
                cache_update_index=cache_update_index,
                output_length=output_length,
                input_length=input_length,
            )
        )

        # Canvas queries see all canvas keys.
        for q in range(2):
            for k in range(4, 8):
                self.assertTrue(mask_np[0, q, k])
        # Non-canvas queries never see canvas keys.
        for q in (2, 3):
            for k in range(4, 8):
                self.assertFalse(mask_np[0, q, k])
        # Non-canvas keys are never attended, even by canvas queries.
        for q in range(2):
            for k in list(range(4)) + list(range(8, 10)):
                self.assertFalse(mask_np[0, q, k])

    def test_serialization(self):
        self.run_serialization_test(self.layer)

    def test_sliding_window_applies_in_both_modes(self):
        """Apply the sliding window in both encoder and decoder modes."""
        sliding_window_size = 6
        layer = DiffusionGemmaTransformerLayer(
            hidden_dim=self.hidden_dim,
            intermediate_dim=16,
            head_dim=self.head_dim,
            num_query_heads=self.num_query_heads,
            num_key_value_heads=self.num_key_value_heads,
            use_sliding_window_attention=True,
            sliding_window_size=sliding_window_size,
            is_global_attention=False,
        )
        x = np.random.randn(1, 9, self.hidden_dim).astype("float32")

        for is_encoder in (True, False):
            mask = layer._compute_attention_mask(
                x,
                padding_mask=None,
                cache=None,
                cache_update_index=0,
                is_encoder=is_encoder,
            )
            mask_np = ops.convert_to_numpy(mask)
            # Every query attends to at most `sliding_window_size` keys,
            # and always to itself — regardless of is_encoder.
            for q in range(9):
                row = mask_np[0, q]
                self.assertTrue(row[q])
                self.assertLessEqual(int(row.sum()), sliding_window_size)

    def test_sliding_window_applies_to_all_queries(self):
        """Apply the sliding window to every query in a multi-token block."""
        sliding_window_size = 6
        layer = DiffusionGemmaTransformerLayer(
            hidden_dim=self.hidden_dim,
            intermediate_dim=16,
            head_dim=self.head_dim,
            num_query_heads=self.num_query_heads,
            num_key_value_heads=self.num_key_value_heads,
            use_sliding_window_attention=True,
            sliding_window_size=sliding_window_size,
            is_global_attention=False,
        )
        # 5 positions of prefix (cache_update_index=5) + 4 query positions,
        # matching _decode_canvas_step's saturated-prefix shape.
        canvas_length = 4
        cache_update_index = 5
        total_length = cache_update_index + canvas_length
        x = np.random.randn(1, canvas_length, self.hidden_dim).astype("float32")
        cache = np.zeros(
            (1, 2, total_length, self.num_key_value_heads, self.head_dim),
            dtype="float32",
        )
        mask = layer._compute_attention_mask(
            x,
            padding_mask=None,
            cache=ops.convert_to_tensor(cache),
            cache_update_index=cache_update_index,
            is_encoder=False,
        )
        mask_np = ops.convert_to_numpy(mask)
        for row_index in range(canvas_length):
            query_position = cache_update_index + row_index
            row = mask_np[0, row_index]
            # Attends only to [query_position - window + 1, query_position].
            expected_start = query_position - sliding_window_size + 1
            for key_position in range(total_length):
                expected = expected_start <= key_position <= query_position
                self.assertEqual(
                    bool(row[key_position]),
                    expected,
                    f"row={row_index} key={key_position}",
                )

    def test_call_auto_slices_local_cache(self):
        """Match internal cache slicing with manual cache slicing."""
        sliding_window_size = 6
        layer = DiffusionGemmaTransformerLayer(
            hidden_dim=self.hidden_dim,
            intermediate_dim=16,
            head_dim=self.head_dim,
            num_query_heads=self.num_query_heads,
            num_key_value_heads=self.num_key_value_heads,
            use_sliding_window_attention=True,
            sliding_window_size=sliding_window_size,
            is_global_attention=False,
        )
        canvas_length = 4
        prompt_length = 3  # < sliding_window_size - 1 = 5.
        total_length = prompt_length + canvas_length
        x = ops.convert_to_tensor(
            np.random.randn(1, canvas_length, self.hidden_dim).astype("float32")
        )
        cache = ops.convert_to_tensor(
            np.random.randn(
                1, 2, total_length, self.num_key_value_heads, self.head_dim
            ).astype("float32")
        )
        padding_mask = ops.ones((1, total_length), dtype="bool")
        canvas_mask = ops.ones((1, canvas_length), dtype="bool")
        layer(x, cache=cache, cache_update_index=0)  # build

        # Auto-sliced: full cache, true prompt_length, layer slices itself.
        auto_out, _ = layer(
            x,
            cache=cache,
            cache_update_index=prompt_length,
            canvas_mask=canvas_mask,
            padding_mask=padding_mask,
            return_cache=False,
        )

        # Manually pre-sliced: mirrors the old _decode_canvas_step logic.
        window_prefix = sliding_window_size - 1
        prefix_length = min(prompt_length, window_prefix)
        cache_start = prompt_length - prefix_length
        local_cache_length = prefix_length + canvas_length
        manual_cache = ops.slice(
            cache,
            (0, 0, cache_start, 0, 0),
            (1, 2, local_cache_length, self.num_key_value_heads, self.head_dim),
        )
        manual_padding_mask = ops.slice(
            padding_mask, (0, cache_start), (1, local_cache_length)
        )
        manual_positions = ops.broadcast_to(
            ops.expand_dims(
                ops.arange(
                    prompt_length,
                    prompt_length + canvas_length,
                    dtype="int32",
                ),
                axis=0,
            ),
            (1, canvas_length),
        )
        manual_out, _ = layer(
            x,
            cache=manual_cache,
            cache_update_index=prefix_length,
            canvas_mask=canvas_mask,
            padding_mask=manual_padding_mask,
            positions=manual_positions,
            return_cache=False,
        )

        self.assertAllClose(
            ops.convert_to_numpy(auto_out),
            ops.convert_to_numpy(manual_out),
            atol=1e-5,
        )

    def test_vision_bidirectional_mask_applies_only_during_encoder_pass(self):
        layer = DiffusionGemmaTransformerLayer(
            hidden_dim=self.hidden_dim,
            intermediate_dim=16,
            head_dim=self.head_dim,
            num_query_heads=self.num_query_heads,
            num_key_value_heads=self.num_key_value_heads,
            use_vision_bidirectional_attention=True,
            is_global_attention=False,
        )
        vision_mask = np.array(
            [[False, False, True, True, True, False, False, False]] * 2,
            dtype=bool,
        )

        encoder_mask = layer._compute_attention_mask(
            self.dummy_input,
            padding_mask=None,
            cache=None,
            cache_update_index=0,
            is_encoder=True,
            vision_mask=vision_mask,
        )
        decoder_mask = layer._compute_attention_mask(
            self.dummy_input,
            padding_mask=None,
            cache=None,
            cache_update_index=0,
            is_encoder=False,
            vision_mask=vision_mask,
        )
        encoder_mask_np = ops.convert_to_numpy(encoder_mask)
        decoder_mask_np = ops.convert_to_numpy(decoder_mask)

        # Encoder pass: image token 2 attends to later image token 4.
        self.assertTrue(encoder_mask_np[0, 2, 4])
        # Decoder pass (is_encoder=False): stays purely causal.
        self.assertFalse(decoder_mask_np[0, 2, 4])

    def test_vision_bidirectional_mask_applies_to_global_layers(self):
        layer = DiffusionGemmaTransformerLayer(
            hidden_dim=self.hidden_dim,
            intermediate_dim=16,
            head_dim=self.head_dim,
            num_query_heads=self.num_query_heads,
            num_key_value_heads=self.num_key_value_heads,
            use_vision_bidirectional_attention=True,
            is_global_attention=True,
        )
        vision_mask = np.array(
            [[False, False, True, True, True, False, False, False]] * 2,
            dtype=bool,
        )
        mask = layer._compute_attention_mask(
            self.dummy_input,
            padding_mask=None,
            cache=None,
            cache_update_index=0,
            is_encoder=True,
            vision_mask=vision_mask,
        )
        mask_np = ops.convert_to_numpy(mask)
        # Global layers also receive the vision-bidirectional mask.
        self.assertTrue(mask_np[0, 2, 4])
