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
            np.array(ops.stop_gradient(out_enc)),
            np.array(ops.stop_gradient(out_dec)),
        )

        # Symmetry: equal scalars → matching outputs.
        self.layer.encoder_layer_scalar.assign(0.5)
        out_enc_equal, _ = self.layer(x, is_encoder=True)
        self.assertAllClose(
            np.array(ops.stop_gradient(out_enc_equal)),
            np.array(ops.stop_gradient(out_dec)),
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
