from unittest.mock import patch

import pytest
from keras import ops

from keras_hub.src.models.smollm3.smollm3_backbone import SmolLM3Backbone
from keras_hub.src.models.smollm3.smollm3_layers import SmolLM3Attention
from keras_hub.src.tests.test_case import TestCase


class SmolLM3BackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 100,
            "hidden_dim": 64,
            "intermediate_dim": 128,
            "num_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "rope_layer_enabled_list": [True, True],
            "layer_types": ["attention", "attention"],
            "mlp_bias": False,
            "layer_norm_epsilon": 1e-5,
            "max_position_embeddings": 128,
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
        }

    def _make_attention(self):
        return SmolLM3Attention(
            hidden_size=8,
            num_attention_heads=2,
            num_key_value_heads=2,
            attention_bias=False,
            attention_dropout=0.0,
            rope_layer_enabled_list=[True],
            layer_types=["attention"],
            layer_idx=0,
        )

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=SmolLM3Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 64),
            run_mixed_precision_check=False,
            run_quantization_check=False,
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=SmolLM3Backbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_num_parameters(self):
        model = SmolLM3Backbone(**self.init_kwargs)
        # Reference value calculated from the model architecture
        self.assertEqual(model.count_params(), 80464)

    def test_fused_attention(self):
        attention = self._make_attention()
        query = ops.ones((1, 4, 2, 4))
        key = ops.ones((1, 4, 2, 4))
        value = ops.ones((1, 4, 2, 4))
        expected = ops.zeros_like(query)

        for attention_mask in (
            None,
            ops.ones((1, 4, 4), dtype="int32"),
        ):
            with (
                self.subTest(attention_mask=attention_mask),
                patch(
                    "keras_hub.src.models.smollm3.smollm3_layers."
                    "fused_attention_op_available",
                    return_value=True,
                ),
                patch.object(
                    ops, "dot_product_attention", return_value=expected
                ) as mock_attention,
            ):
                output = attention._compute_attention(
                    query, key, value, attention_mask
                )

                self.assertIs(output, expected)
                mock_attention.assert_called_once()
                args, kwargs = mock_attention.call_args
                self.assertIs(args[0], query)
                self.assertIs(args[1], key)
                self.assertIs(args[2], value)
                expected_mask = (
                    None
                    if attention_mask is None
                    else ops.expand_dims(
                        ops.cast(attention_mask, "bool"), axis=1
                    )
                )
                self.assertAllEqual(kwargs["mask"], expected_mask)
                self.assertEqual(kwargs["scale"], attention._inv_norm_factor)

    def test_fused_attention_fallback(self):
        attention = self._make_attention()
        query = ops.ones((1, 4, 2, 4))
        key = ops.ones((1, 4, 2, 4))
        value = ops.ones((1, 4, 2, 4))
        attention_mask = ops.ones((1, 4, 4), dtype="bool")

        with (
            patch(
                "keras_hub.src.models.smollm3.smollm3_layers."
                "fused_attention_op_available",
                return_value=False,
            ),
            patch.object(ops, "dot_product_attention") as mock_attention,
        ):
            output = attention._compute_attention(
                query, key, value, attention_mask
            )

        mock_attention.assert_not_called()
        self.assertAllClose(output, ops.ones_like(output))

    def test_fused_attention_matches_fallback(self):
        attention = self._make_attention()
        query = ops.reshape(ops.arange(32, dtype="float32"), (1, 4, 2, 4))
        key = ops.flip(query, axis=1)
        value = ops.cos(query)
        attention_mask = ops.tril(ops.ones((1, 4, 4), dtype="bool"))

        with patch(
            "keras_hub.src.models.smollm3.smollm3_layers."
            "fused_attention_op_available",
            return_value=False,
        ):
            fallback_output = attention._compute_attention(
                query, key, value, attention_mask
            )
        with patch(
            "keras_hub.src.models.smollm3.smollm3_layers."
            "fused_attention_op_available",
            return_value=True,
        ):
            fused_output = attention._compute_attention(
                query, key, value, attention_mask
            )

        self.assertAllClose(fused_output, fallback_output, atol=1e-5)
