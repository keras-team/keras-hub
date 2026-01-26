"""Tests for Qwen3-Omni backbone."""

import pytest
from keras import ops

from keras_hub.src.models.qwen3_omni.qwen3_omni_backbone import (
    Qwen3OmniBackbone,
)
from keras_hub.src.tests.test_case import TestCase


class Qwen3OmniBackboneTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 100,
            "num_layers": 2,
            "num_query_heads": 8,
            "num_key_value_heads": 2,
            "hidden_dim": 64,  # Reduced to match head_dim * num_query_heads
            "intermediate_dim": 32,
            "head_dim": 8,  # Reduced for faster tests
            "moe_intermediate_dim": 32,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "norm_topk_prob": True,
            "decoder_sparse_step": 1,
            "mrope_section": (2, 1, 1),  # Must sum to head_dim // 2 = 4
            "layer_norm_epsilon": 1e-6,
            "rope_max_wavelength": 1000000,
            "rope_scaling_factor": 1.0,
            "dropout": 0.0,
            "tie_word_embeddings": False,
            "sliding_window_size": None,
            "router_aux_loss_coefficient": 0.001,
            "mlp_only_layers": [],
            "dtype": "float32",
        }
        self.input_data = {
            "token_ids": ops.ones((2, 8), dtype="int32"),
            "padding_mask": ops.ones((2, 8), dtype="int32"),
        }

    def test_backbone_basics(self):
        """Test basic backbone instantiation and forward pass."""
        self.run_backbone_test(
            cls=Qwen3OmniBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 8, 64),
            run_quantization_check=False,
        )

    @pytest.mark.large
    def test_saved_model(self):
        """Test model serialization and deserialization."""
        self.run_model_saving_test(
            cls=Qwen3OmniBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_architecture_characteristics(self):
        """Test expected number of parameters and layers."""
        model = Qwen3OmniBackbone(**self.init_kwargs)
        # Verify model can be instantiated
        self.assertIsNotNone(model)
        # Verify it has transformer layers
        self.assertEqual(len(model.transformer_layers), 2)
        # Verify non-zero parameters
        self.assertGreater(model.count_params(), 0)

    def test_auxiliary_loss(self):
        """Test MoE auxiliary loss is added during training."""
        model = Qwen3OmniBackbone(**self.init_kwargs)
        _ = model(self.input_data, training=True)
        # MoE layers should add auxiliary losses for load balancing
        self.assertTrue(
            len(model.losses) > 0, "Auxiliary losses should be present"
        )
        for loss in model.losses:
            self.assertGreater(loss, 0.0, "Auxiliary loss should be positive")

    def test_no_auxiliary_loss_inference(self):
        """Test no auxiliary loss is added during inference."""
        model = Qwen3OmniBackbone(**self.init_kwargs)
        _ = model(self.input_data, training=False)
        # No auxiliary loss should be added during inference
        self.assertEqual(
            len(model.losses), 0, "No auxiliary losses during inference"
        )

    def test_mlp_only_layers(self):
        """Test layers can use dense FFN instead of MoE."""
        init_kwargs = self.init_kwargs.copy()
        init_kwargs["mlp_only_layers"] = [0]  # First layer uses dense FFN
        model = Qwen3OmniBackbone(**init_kwargs)
        # Should still work with mixed sparse/dense layers
        output = model(self.input_data)
        self.assertEqual(ops.shape(output), (2, 8, 64))

    def test_config_serialization(self):
        """Test model config can be serialized and deserialized."""
        model = Qwen3OmniBackbone(**self.init_kwargs)
        config = model.get_config()
        
        # Verify all important config keys are present
        self.assertIn("vocabulary_size", config)
        self.assertIn("num_layers", config)
        self.assertIn("num_experts", config)
        self.assertIn("num_experts_per_tok", config)
        self.assertIn("rope_max_wavelength", config)
        
        # Test config values match
        self.assertEqual(config["vocabulary_size"], 100)
        self.assertEqual(config["num_layers"], 2)
        self.assertEqual(config["num_experts"], 8)
        self.assertEqual(config["rope_max_wavelength"], 1000000)

    def test_from_config(self):
        """Test model can be reconstructed from config."""
        model = Qwen3OmniBackbone(**self.init_kwargs)
        config = model.get_config()
        
        # Reconstruct model from config
        new_model = Qwen3OmniBackbone.from_config(config)
        
        # Verify same architecture
        self.assertEqual(model.count_params(), new_model.count_params())
        self.assertEqual(len(model.layers), len(new_model.layers))

    def test_sliding_window(self):
        """Test sliding window attention."""
        init_kwargs = self.init_kwargs.copy()
        init_kwargs["sliding_window_size"] = 4
        model = Qwen3OmniBackbone(**init_kwargs)
        
        # Should work with sliding window
        output = model(self.input_data)
        self.assertEqual(ops.shape(output), (2, 8, 64))

    def test_token_embedding(self):
        """Test token embeddings are properly initialized."""
        model = Qwen3OmniBackbone(**self.init_kwargs)
        # Verify embedding layer exists
        self.assertIsNotNone(model.token_embedding)
        # Verify embedding dimensions
        self.assertEqual(model.token_embedding.input_dim, 100)
        self.assertEqual(model.token_embedding.output_dim, 64)

    def test_multimodal_encoders_none(self):
        """Test model works without multimodal encoders (text-only mode)."""
        init_kwargs = self.init_kwargs.copy()
        init_kwargs["audio_encoder"] = None
        init_kwargs["vision_encoder"] = None
        model = Qwen3OmniBackbone(**init_kwargs)
        
        # Text-only forward pass should work
        output = model(self.input_data)
        self.assertEqual(ops.shape(output), (2, 8, 64))

    def test_different_batch_sizes(self):
        """Test model works with different batch sizes."""
        model = Qwen3OmniBackbone(**self.init_kwargs)
        
        # Test batch size 1
        input_data_batch1 = {
            "token_ids": ops.ones((1, 8), dtype="int32"),
            "padding_mask": ops.ones((1, 8), dtype="int32"),
        }
        output1 = model(input_data_batch1)
        self.assertEqual(ops.shape(output1), (1, 8, 64))
        
        # Test batch size 4
        input_data_batch4 = {
            "token_ids": ops.ones((4, 8), dtype="int32"),
            "padding_mask": ops.ones((4, 8), dtype="int32"),
        }
        output4 = model(input_data_batch4)
        self.assertEqual(ops.shape(output4), (4, 8, 64))

    def test_different_sequence_lengths(self):
        """Test model works with different sequence lengths."""
        model = Qwen3OmniBackbone(**self.init_kwargs)
        
        # Test shorter sequence
        input_data_short = {
            "token_ids": ops.ones((2, 4), dtype="int32"),
            "padding_mask": ops.ones((2, 4), dtype="int32"),
        }
        output_short = model(input_data_short)
        self.assertEqual(ops.shape(output_short), (2, 4, 64))
        
        # Test longer sequence
        input_data_long = {
            "token_ids": ops.ones((2, 16), dtype="int32"),
            "padding_mask": ops.ones((2, 16), dtype="int32"),
        }
        output_long = model(input_data_long)
        self.assertEqual(ops.shape(output_long), (2, 16, 64))
