import pytest
from keras import ops

from keras_hub.src.models.qwen3_omni_moe.qwen3_omni_moe_backbone import Qwen3OmniMoeBackbone
from keras_hub.src.tests.test_case import TestCase


class Qwen3OmniMoeBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 151936,
            "num_layers": 2,
            "num_query_heads": 4,
            "num_key_value_heads": 2,
            "hidden_dim": 128,
            "intermediate_dim": 256,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "head_dim": 32,
            "max_sequence_length": 128,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 16), dtype="int32"),
            "padding_mask": ops.ones((2, 16), dtype="int32"),
        }

    def test_backbone_basics(self):
        self.run_backbone_test(
            cls=Qwen3OmniMoeBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 16, 128),
            run_mixed_precision_check=False,  # Disable mixed precision check due to MoE complexity
            run_quantization_check=False,     # Disable quantization check due to MoE complexity
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=Qwen3OmniMoeBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in Qwen3OmniMoeBackbone.presets:
            self.run_preset_test(
                cls=Qwen3OmniMoeBackbone,
                preset=preset,
                input_data=self.input_data,
            )

    def test_output_shapes(self):
        """Test that output shapes are correct."""
        model = Qwen3OmniMoeBackbone(**self.init_kwargs)
        
        # Test with different sequence lengths
        for seq_len in [16, 32]:
            test_input = {
                "token_ids": ops.ones((2, seq_len), dtype="int32"),
                "padding_mask": ops.ones((2, seq_len), dtype="int32"),
            }
            outputs = model(test_input)
            self.assertEqual(outputs.shape, (2, seq_len, 128))

    def test_auxiliary_loss(self):
        """Test that auxiliary losses are properly computed during training."""
        model = Qwen3OmniMoeBackbone(**self.init_kwargs)
        _ = model(self.input_data, training=True)
        self.assertTrue(
            len(model.losses) > 0, "Auxiliary losses should be present"
        )
        for loss in model.losses:
            self.assertGreater(loss, 0.0, "Auxiliary loss should be positive")