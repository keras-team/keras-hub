import pytest
from keras import ops

from keras_hub.src.models.modernbert.modern_bert_backbone import (
    ModernBertBackbone,
)
from keras_hub.src.tests.test_case import TestCase


class ModernBertBackboneTest(TestCase):
    """Tests for ModernBERT backbone."""

    def setUp(self):
        """Set up a small configuration for testing."""
        self.init_kwargs = {
            "vocabulary_size": 10,
            "hidden_dim": 8,
            "intermediate_dim": 32,
            "num_layers": 2,
            "num_heads": 4,
            "local_attention_window": 128,
            "global_attn_every_n_layers": 2,
            "dropout": 0.0,
        }

        self.input_data = {
            "token_ids": ops.ones(
                (2, 5),
                dtype="int32",
            ),
            "padding_mask": ops.ones(
                (2, 5),
                dtype="int32",
            ),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=ModernBertBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 8),
        )

    def test_variable_sequence_length(self):
        model = ModernBertBackbone(**self.init_kwargs)

        output = model(
            {
                "token_ids": ops.ones(
                    (1, 3),
                    dtype="int32",
                ),
                "padding_mask": ops.ones(
                    (1, 3),
                    dtype="int32",
                ),
            }
        )

        self.assertEqual(
            output.shape,
            (1, 3, 8),
        )

    def test_alternating_attention_logic(self):
        """Validate global and local attention layer assignment."""
        model = ModernBertBackbone(**self.init_kwargs)

        self.assertIsNone(model.transformer_layers[0].local_attention_window)

        self.assertEqual(
            model.transformer_layers[1].local_attention_window,
            128,
        )

    def test_serialization(self):
        model = ModernBertBackbone(**self.init_kwargs)
        self.run_serialization_test(
            model,
        )

    @pytest.mark.extra_large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=ModernBertBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_mixed_precision(self):
        self.run_precision_test(
            cls=ModernBertBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 8),
        )

    @pytest.mark.extra_large
    def test_smallest_preset(self):
        self.run_preset_test(
            cls=ModernBertBackbone,
            preset="modernbert_base_en",
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in ModernBertBackbone.presets:
            self.run_preset_test(
                cls=ModernBertBackbone,
                preset=preset,
                input_data=self.input_data,
            )
