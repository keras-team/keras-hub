import numpy as np
from keras import ops
from keras_hub.src.tests.test_case import TestCase
from modernbert_layers import (
    ModernBertMLP, ModernBertAttention, ModernBertEncoderLayer,
)

class ModernBertLayersTest(TestCase):
    def test_mlp_forward(self):
        layer = ModernBertMLP(hidden_dim=16, intermediate_dim=32)
        x = ops.ones((2, 5, 16))
        self.assertAllEqual(layer(x).shape, (2, 5, 16))

    def test_attention_masking(self):
        layer = ModernBertAttention(hidden_dim=16, num_heads=2)
        x = ops.ones((1, 4, 16))
        # Mask out last two tokens (1 is valid, 0 is padding)
        mask = ops.convert_to_tensor([[1, 1, 0, 0]], dtype="int32")
        output = layer(x, padding_mask=mask)
        self.assertFalse(np.any(np.isnan(output.numpy())))

    def test_serialization(self):
        layer = ModernBertEncoderLayer(
            hidden_dim=16, intermediate_dim=32, num_heads=2, local_attention_window=64
        )
        config = layer.get_config()
        new_layer = ModernBertEncoderLayer.from_config(config)
        self.assertEqual(new_layer.local_attention_window, 64)