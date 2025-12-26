"""Tests for Llama3VisionCrossAttention."""

import keras
import numpy as np

from keras_hub.src.models.llama3.llama3_vision_cross_attention import (
    Llama3VisionCrossAttention,
)
from keras_hub.src.tests.test_case import TestCase


class Llama3VisionCrossAttentionTest(TestCase):
    """Test cases for Llama3VisionCrossAttention layer."""

    def test_output_shape(self):
        """Test that output shape matches input hidden states shape."""
        layer = Llama3VisionCrossAttention(
            hidden_dim=256,
            num_heads=8,
            num_key_value_heads=4,
        )

        batch_size = 2
        seq_len = 16
        num_patches = 64
        hidden_dim = 256

        hidden_states = np.random.randn(batch_size, seq_len, hidden_dim).astype(
            np.float32
        )
        vision_features = np.random.randn(
            batch_size, num_patches, hidden_dim
        ).astype(np.float32)

        output = layer(hidden_states, vision_features)

        assert output.shape == (batch_size, seq_len, hidden_dim)

    def test_gate_initialization(self):
        """Test that gate is initialized to zero."""
        layer = Llama3VisionCrossAttention(
            hidden_dim=256,
            num_heads=8,
        )

        # Build the layer
        hidden_states = np.random.randn(2, 16, 256).astype(np.float32)
        vision_features = np.random.randn(2, 64, 256).astype(np.float32)
        layer(hidden_states, vision_features)

        # Check gate value
        gate_value = layer.gate.numpy()
        np.testing.assert_allclose(gate_value, [0.0], atol=1e-6)

    def test_zero_gate_passthrough(self):
        """Test that with zero gate, output equals input."""
        layer = Llama3VisionCrossAttention(
            hidden_dim=256,
            num_heads=8,
        )

        hidden_states = np.random.randn(2, 16, 256).astype(np.float32)
        vision_features = np.random.randn(2, 64, 256).astype(np.float32)

        output = layer(hidden_states, vision_features)

        # With gate=0, tanh(0)=0, so output should equal hidden_states
        np.testing.assert_allclose(output.numpy(), hidden_states, atol=1e-5)

    def test_with_vision_mask(self):
        """Test that vision mask works correctly."""
        layer = Llama3VisionCrossAttention(
            hidden_dim=256,
            num_heads=8,
        )

        batch_size = 2
        seq_len = 16
        num_patches = 64
        hidden_dim = 256

        hidden_states = np.random.randn(batch_size, seq_len, hidden_dim).astype(
            np.float32
        )
        vision_features = np.random.randn(
            batch_size, num_patches, hidden_dim
        ).astype(np.float32)
        # Create a mask where first half are visible
        vision_mask = np.ones((batch_size, num_patches), dtype=bool)
        vision_mask[:, num_patches // 2 :] = False

        output = layer(hidden_states, vision_features, vision_mask=vision_mask)

        assert output.shape == (batch_size, seq_len, hidden_dim)

    def test_standard_mha(self):
        """Test with equal num_heads and num_key_value_heads (standard MHA)."""
        layer = Llama3VisionCrossAttention(
            hidden_dim=256,
            num_heads=8,
            num_key_value_heads=8,  # Same as num_heads
        )

        hidden_states = np.random.randn(2, 16, 256).astype(np.float32)
        vision_features = np.random.randn(2, 64, 256).astype(np.float32)

        output = layer(hidden_states, vision_features)
        assert output.shape == (2, 16, 256)

    def test_gqa(self):
        """Test with grouped query attention (fewer kv heads)."""
        layer = Llama3VisionCrossAttention(
            hidden_dim=256,
            num_heads=8,
            num_key_value_heads=2,  # GQA with 4 groups
        )

        hidden_states = np.random.randn(2, 16, 256).astype(np.float32)
        vision_features = np.random.randn(2, 64, 256).astype(np.float32)

        output = layer(hidden_states, vision_features)
        assert output.shape == (2, 16, 256)

    def test_serialization(self):
        """Test that the layer can be serialized and deserialized."""
        layer = Llama3VisionCrossAttention(
            hidden_dim=256,
            num_heads=8,
            num_key_value_heads=4,
            layer_norm_epsilon=1e-5,
            dropout=0.1,
        )

        # Build the layer
        hidden_states = np.random.randn(2, 16, 256).astype(np.float32)
        vision_features = np.random.randn(2, 64, 256).astype(np.float32)
        layer(hidden_states, vision_features)

        config = layer.get_config()

        # Recreate from config
        new_layer = Llama3VisionCrossAttention(**config)

        assert new_layer.hidden_dim == layer.hidden_dim
        assert new_layer.num_heads == layer.num_heads
        assert new_layer.num_key_value_heads == layer.num_key_value_heads

    def test_get_config(self):
        """Test get_config returns correct configuration."""
        layer = Llama3VisionCrossAttention(
            hidden_dim=512,
            num_heads=16,
            num_key_value_heads=8,
            layer_norm_epsilon=1e-6,
            dropout=0.2,
        )

        config = layer.get_config()

        assert config["hidden_dim"] == 512
        assert config["num_heads"] == 16
        assert config["num_key_value_heads"] == 8
        assert config["layer_norm_epsilon"] == 1e-6
        assert config["dropout"] == 0.2

    def test_gradient_flow(self):
        """Test that gradients flow properly through the layer."""
        import tensorflow as tf
        
        layer = Llama3VisionCrossAttention(
            hidden_dim=256,
            num_heads=8,
        )

        hidden_states = keras.Variable(
            np.random.randn(2, 16, 256).astype(np.float32)
        )
        vision_features = keras.Variable(
            np.random.randn(2, 64, 256).astype(np.float32)
        )

        # Test gradient flow
        with tf.GradientTape() as tape:
            output = layer(hidden_states, vision_features)
            loss = tf.reduce_sum(output)

        # Check gradients exist
        grads = tape.gradient(loss, layer.trainable_weights)
        for grad in grads:
            assert grad is not None
