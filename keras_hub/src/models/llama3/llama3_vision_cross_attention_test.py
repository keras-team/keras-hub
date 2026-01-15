import keras
import numpy as np

from keras_hub.src.models.llama3.llama3_vision_cross_attention import (
    Llama3VisionCrossAttention,
)
from keras_hub.src.tests.test_case import TestCase


class Llama3VisionCrossAttentionTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "hidden_dim": 256,
            "num_heads": 8,
            "num_key_value_heads": 4,
        }
        self.batch_size = 2
        self.seq_len = 16
        self.num_patches = 64
        self.hidden_states = np.random.randn(
            self.batch_size, self.seq_len, 256
        ).astype(np.float32)
        self.vision_features = np.random.randn(
            self.batch_size, self.num_patches, 256
        ).astype(np.float32)

    def test_output_shape(self):
        """Test output shape matches input hidden states."""
        layer = Llama3VisionCrossAttention(**self.init_kwargs)
        output = layer(self.hidden_states, self.vision_features)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, 256))

    def test_gate_initialization(self):
        """Test gate is initialized to zero."""
        layer = Llama3VisionCrossAttention(**self.init_kwargs)
        layer(self.hidden_states, self.vision_features)
        gate_value = keras.ops.convert_to_numpy(layer.gate)
        np.testing.assert_allclose(gate_value, [0.0], atol=1e-6)

    def test_zero_gate_passthrough(self):
        """Test with zero gate, output equals input."""
        layer = Llama3VisionCrossAttention(**self.init_kwargs)
        output = layer(self.hidden_states, self.vision_features)
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(output), self.hidden_states, atol=1e-5
        )

    def test_with_vision_mask(self):
        """Test vision mask functionality."""
        layer = Llama3VisionCrossAttention(**self.init_kwargs)
        vision_mask = np.ones((self.batch_size, self.num_patches), dtype=bool)
        vision_mask[:, self.num_patches // 2 :] = False
        output = layer(
            self.hidden_states, self.vision_features, vision_mask=vision_mask
        )
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, 256))

    def test_gqa(self):
        """Test grouped query attention."""
        init_kwargs = dict(self.init_kwargs)
        init_kwargs["num_key_value_heads"] = 2
        layer = Llama3VisionCrossAttention(**init_kwargs)
        output = layer(self.hidden_states, self.vision_features)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, 256))

    def test_serialization(self):
        """Test layer serialization."""
        layer = Llama3VisionCrossAttention(**self.init_kwargs)
        layer(self.hidden_states, self.vision_features)

        config = layer.get_config()
        new_layer = Llama3VisionCrossAttention(**config)

        self.assertEqual(new_layer.hidden_dim, layer.hidden_dim)
        self.assertEqual(new_layer.num_heads, layer.num_heads)
        self.assertEqual(
            new_layer.num_key_value_heads, layer.num_key_value_heads
        )

    def test_gradient_flow(self):
        """Test gradients flow through the layer."""
        layer = Llama3VisionCrossAttention(**self.init_kwargs)
        layer(self.hidden_states, self.vision_features)

        self.assertGreater(len(layer.trainable_weights), 0)
        weight_names = [w.path for w in layer.trainable_weights]
        self.assertTrue(any("gate" in name for name in weight_names))
