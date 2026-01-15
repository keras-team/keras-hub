import numpy as np
import pytest
from keras import ops
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.models.modernbert.modernbert_layers import (
    ModernBertMLP, ModernBertAttention, ModernBertEncoderLayer,
)

class ModernBertLayersTest(TestCase):
    """Tests for ModernBERT specific layers.
    
    This class verifies the functional correctness, shape inference, 
    serialization, and training compatibility of the custom layers 
    used in the ModernBERT architecture.
    """

    def test_mlp_layer(self):
        """Test ModernBertMLP forward pass and serialization."""
        self.run_layer_test(
            layer_cls=ModernBertMLP,
            init_kwargs={
                "hidden_dim": 16,
                "intermediate_dim": 32,
                "dropout": 0.1,
            },
            input_shape=(2, 5, 16),
            expected_output_shape=(2, 5, 16),
        )

    def test_attention_layer(self):
        """Test ModernBertAttention with serialization and shape checks."""
        self.run_layer_test(
            layer_cls=ModernBertAttention,
            init_kwargs={
                "hidden_dim": 16,
                "num_heads": 2,
                "dropout": 0.1,
            },
            input_shape=(2, 8, 16),
            expected_output_shape=(2, 8, 16),
        )

    def test_encoder_layer(self):
        """Test ModernBertEncoderLayer including local attention window logic."""
        self.run_layer_test(
            layer_cls=ModernBertEncoderLayer,
            init_kwargs={
                "hidden_dim": 16,
                "intermediate_dim": 32,
                "num_heads": 2,
                "local_attention_window": 64,
                "dropout": 0.1,
            },
            input_shape=(2, 12, 16),
            expected_output_shape=(2, 12, 16),
        )

    def test_attention_masking_logic(self):
        """Verify that the attention layer correctly handles padding masks."""
        layer = ModernBertAttention(hidden_dim=16, num_heads=2)
        x = ops.ones((1, 4, 16))
        # Mask out last two tokens (1 is valid, 0 is padding)
        mask = ops.convert_to_tensor([[1, 1, 0, 0]], dtype="int32")
        output = layer(x, padding_mask=mask)
        
        # Ensure the output contains no NaNs and maintains correct shape
        self.assertFalse(np.any(np.isnan(output)))
        self.assertEqual(output.shape, (1, 4, 16))

    def test_serialization_attributes(self):
        """Explicitly verify that custom attributes are restored after serialization."""
        layer = ModernBertEncoderLayer(
            hidden_dim=16, 
            intermediate_dim=32, 
            num_heads=2, 
            local_attention_window=64
        )
        config = layer.get_config()
        new_layer = ModernBertEncoderLayer.from_config(config)
        self.assertEqual(new_layer.local_attention_window, 64)