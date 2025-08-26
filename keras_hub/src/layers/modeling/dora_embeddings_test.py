"""Test suite for DoRA Embedding Layer Implementation.

This module contains comprehensive tests for the
DoRAEmbedding and DoRAPositionEmbedding
layers, including functionality, compatibility, and edge cases.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import layers

# Import the modules to test
from .dora_embeddings import DoRAEmbedding
from .dora_embeddings import DoRAPositionEmbedding
from .dora_embeddings import convert_embedding_to_dora


class TestDoRAEmbedding:
    """Test class for DoRAEmbedding layer."""

    def setup_method(self):
        """Set up test fixtures."""
        keras.backend.clear_session()
        np.random.seed(42)
        tf.random.set_seed(42)

    def test_init_valid_params(self):
        """Test DoRAEmbedding initialization with valid parameters."""
        layer = DoRAEmbedding(
            input_dim=1000,
            output_dim=128,
            rank=16,
            alpha=2.0,
            mask_zero=True,
            sparse=False,
        )

        assert layer.input_dim == 1000
        assert layer.output_dim == 128
        assert layer.rank == 16
        assert layer.alpha == 2.0
        assert layer.mask_zero is True
        assert layer.sparse is False
        assert layer.scaling == 2.0 / 16  # alpha / rank

    def test_init_invalid_params(self):
        """Test DoRAEmbedding initialization with invalid parameters."""
        # Test invalid input_dim
        with pytest.raises(ValueError, match="input_dim must be positive"):
            DoRAEmbedding(input_dim=0, output_dim=128)

        with pytest.raises(ValueError, match="input_dim must be positive"):
            DoRAEmbedding(input_dim=-10, output_dim=128)

        # Test invalid output_dim
        with pytest.raises(ValueError, match="output_dim must be positive"):
            DoRAEmbedding(input_dim=1000, output_dim=0)

        with pytest.raises(ValueError, match="output_dim must be positive"):
            DoRAEmbedding(input_dim=1000, output_dim=-5)

        # Test invalid rank
        with pytest.raises(ValueError, match="rank must be positive"):
            DoRAEmbedding(input_dim=1000, output_dim=128, rank=0)

        with pytest.raises(ValueError, match="rank must be positive"):
            DoRAEmbedding(input_dim=1000, output_dim=128, rank=-4)

        # Test invalid alpha
        with pytest.raises(ValueError, match="alpha must be positive"):
            DoRAEmbedding(input_dim=1000, output_dim=128, alpha=0)

        with pytest.raises(ValueError, match="alpha must be positive"):
            DoRAEmbedding(input_dim=1000, output_dim=128, alpha=-1.0)

    def test_build(self):
        """Test layer building process."""
        layer = DoRAEmbedding(input_dim=100, output_dim=32, rank=8)
        layer.build(None)  # Embedding layers don't need input shape

        # Check that weights are created
        assert layer.embeddings is not None
        assert layer.lora_a is not None
        assert layer.lora_b is not None
        assert layer.magnitude is not None

        # Check weight shapes
        assert layer.embeddings.shape == (100, 32)
        assert layer.lora_a.shape == (100, 8)
        assert layer.lora_b.shape == (8, 32)
        assert layer.magnitude.shape == (32,)

        # Check trainability
        assert not layer.embeddings.trainable  # Frozen
        assert layer.lora_a.trainable
        assert layer.lora_b.trainable
        assert layer.magnitude.trainable

    def test_call_basic(self):
        """Test basic forward pass."""
        layer = DoRAEmbedding(input_dim=50, output_dim=16, rank=4)
        layer.build(None)

        # Create integer inputs (token indices)
        inputs = np.array([[1, 5, 10, 3], [7, 2, 9, 4]], dtype=np.int32)

        outputs = layer(inputs)

        assert outputs.shape == (2, 4, 16)  # (batch_size, seq_len, output_dim)
        assert outputs.dtype == layer.embeddings.dtype

    def test_call_with_different_dtypes(self):
        """Test forward pass with different input dtypes."""
        layer = DoRAEmbedding(input_dim=20, output_dim=8, rank=2)
        layer.build(None)

        # Test with float inputs (should be cast to int32)
        inputs_float = np.array([[1.0, 5.0], [7.0, 2.0]], dtype=np.float32)
        outputs = layer(inputs_float)
        assert outputs.shape == (2, 2, 8)

        # Test with int64 inputs
        inputs_int64 = np.array([[1, 5], [7, 2]], dtype=np.int64)
        outputs = layer(inputs_int64)
        assert outputs.shape == (2, 2, 8)

    def test_masking(self):
        """Test masking functionality."""
        # Test with mask_zero=True
        layer = DoRAEmbedding(
            input_dim=10, output_dim=4, rank=2, mask_zero=True
        )
        layer.build(None)

        inputs = np.array([[1, 2, 0], [3, 0, 4]], dtype=np.int32)

        # Test mask computation
        mask = layer.compute_mask(inputs)
        expected_mask = np.array([[True, True, False], [True, False, True]])
        np.testing.assert_array_equal(mask.numpy(), expected_mask)

        # Test with mask_zero=False
        layer_no_mask = DoRAEmbedding(
            input_dim=10, output_dim=4, rank=2, mask_zero=False
        )
        layer_no_mask.build(None)

        mask = layer_no_mask.compute_mask(inputs)
        assert mask is None

    def test_get_effective_embeddings(self):
        """Test computing effective embedding matrix."""
        layer = DoRAEmbedding(input_dim=5, output_dim=3, rank=2)
        layer.build(None)

        effective_embeddings = layer.get_effective_embeddings()

        assert effective_embeddings.shape == (5, 3)

        # Should be different from original embeddings due to DoRA adaptation
        assert not np.allclose(
            effective_embeddings.numpy(), layer.embeddings.numpy()
        )

    def test_get_dora_parameters(self):
        """Test getting DoRA parameters."""
        layer = DoRAEmbedding(input_dim=10, output_dim=6, rank=3)
        layer.build(None)

        params = layer.get_dora_parameters()

        assert "lora_a" in params
        assert "lora_b" in params
        assert "magnitude" in params

        assert params["lora_a"] is layer.lora_a
        assert params["lora_b"] is layer.lora_b
        assert params["magnitude"] is layer.magnitude

    def test_merge_weights(self):
        """Test merging DoRA weights."""
        layer = DoRAEmbedding(input_dim=8, output_dim=4, rank=2)
        layer.build(None)

        merged = layer.merge_weights()

        assert "embeddings" in merged
        assert merged["embeddings"].shape == (8, 4)

    def test_count_params(self):
        """Test parameter counting."""
        layer = DoRAEmbedding(input_dim=100, output_dim=50, rank=8)
        layer.build(None)

        expected_params = (
            100 * 8  # lora_a: input_dim * rank
            + 8 * 50  # lora_b: rank * output_dim
            + 50  # magnitude: output_dim
        )
        assert layer.count_params() == expected_params

    def test_load_pretrained_embeddings(self):
        """Test loading pretrained embeddings."""
        layer = DoRAEmbedding(input_dim=6, output_dim=4, rank=2)
        layer.build(None)

        # Create pretrained embeddings
        pretrained_embeddings = np.random.randn(6, 4).astype(np.float32)

        # Store original values
        original_embeddings = layer.embeddings.numpy().copy()

        # Load pretrained embeddings
        layer.load_pretrained_embeddings(pretrained_embeddings)

        # Check that embeddings changed
        np.testing.assert_array_equal(
            layer.embeddings.numpy(), pretrained_embeddings
        )
        assert not np.allclose(layer.embeddings.numpy(), original_embeddings)

    def test_load_pretrained_embeddings_shape_mismatch(self):
        """Test loading pretrained embeddings with wrong shape."""
        layer = DoRAEmbedding(input_dim=6, output_dim=4, rank=2)
        layer.build(None)

        # Wrong shape
        wrong_embeddings = np.random.randn(5, 4).astype(np.float32)
        with pytest.raises(ValueError, match="doesn't match expected shape"):
            layer.load_pretrained_embeddings(wrong_embeddings)

    def test_expand_vocabulary(self):
        """Test vocabulary expansion functionality."""
        layer = DoRAEmbedding(input_dim=10, output_dim=8, rank=4)
        layer.build(None)

        # Expand vocabulary
        expanded_layer = layer.expand_vocabulary(15)

        # Check new layer properties
        assert expanded_layer.input_dim == 15
        assert expanded_layer.output_dim == 8
        assert expanded_layer.rank == 4

        # Check weight shapes
        assert expanded_layer.embeddings.shape == (15, 8)
        assert expanded_layer.lora_a.shape == (15, 4)
        assert expanded_layer.lora_b.shape == (4, 8)
        assert expanded_layer.magnitude.shape == (8,)

        # Check that original weights are preserved
        np.testing.assert_array_equal(
            expanded_layer.embeddings.numpy()[:10], layer.embeddings.numpy()
        )
        np.testing.assert_array_equal(
            expanded_layer.lora_a.numpy()[:10], layer.lora_a.numpy()
        )
        np.testing.assert_array_equal(
            expanded_layer.lora_b.numpy(), layer.lora_b.numpy()
        )
        np.testing.assert_array_equal(
            expanded_layer.magnitude.numpy(), layer.magnitude.numpy()
        )

    def test_expand_vocabulary_with_custom_embeddings(self):
        """Test vocabulary expansion with custom new token embeddings."""
        layer = DoRAEmbedding(input_dim=5, output_dim=4, rank=2)
        layer.build(None)

        # Custom embeddings for new tokens
        new_token_embeddings = np.random.randn(3, 4).astype(np.float32)

        expanded_layer = layer.expand_vocabulary(8, new_token_embeddings)

        # Check that custom embeddings are used
        np.testing.assert_array_equal(
            expanded_layer.embeddings.numpy()[5:], new_token_embeddings
        )

    def test_expand_vocabulary_invalid_params(self):
        """Test vocabulary expansion with invalid parameters."""
        layer = DoRAEmbedding(input_dim=10, output_dim=8, rank=4)
        layer.build(None)

        # Test with smaller vocabulary
        with pytest.raises(ValueError, match="must be greater than current"):
            layer.expand_vocabulary(8)

        # Test with unbuilt layer
        unbuilt_layer = DoRAEmbedding(input_dim=10, output_dim=8, rank=4)
        with pytest.raises(ValueError, match="must be built before expanding"):
            unbuilt_layer.expand_vocabulary(15)

        # Test with wrong shape for new embeddings
        wrong_embeddings = np.random.randn(3, 6).astype(np.float32)
        with pytest.raises(ValueError, match="doesn't match expected shape"):
            layer.expand_vocabulary(13, wrong_embeddings)

    def test_get_config(self):
        """Test layer configuration serialization."""
        layer = DoRAEmbedding(
            input_dim=1000,
            output_dim=128,
            rank=16,
            alpha=2.0,
            mask_zero=True,
            input_length=100,
            sparse=False,
        )

        config = layer.get_config()

        assert config["input_dim"] == 1000
        assert config["output_dim"] == 128
        assert config["rank"] == 16
        assert config["alpha"] == 2.0
        assert config["mask_zero"] is True
        assert config["input_length"] == 100
        assert config["sparse"] is False

    def test_from_config(self):
        """Test layer creation from configuration."""
        original_layer = DoRAEmbedding(
            input_dim=500, output_dim=64, rank=8, alpha=1.5
        )
        config = original_layer.get_config()

        new_layer = DoRAEmbedding.from_config(config)

        assert new_layer.input_dim == original_layer.input_dim
        assert new_layer.output_dim == original_layer.output_dim
        assert new_layer.rank == original_layer.rank
        assert new_layer.alpha == original_layer.alpha

    def test_compute_output_shape(self):
        """Test output shape computation."""
        layer = DoRAEmbedding(input_dim=100, output_dim=32)

        output_shape = layer.compute_output_shape((None, 10))
        assert output_shape == (None, 10, 32)

        output_shape = layer.compute_output_shape((32, 15))
        assert output_shape == (32, 15, 32)

    def test_mathematical_correctness(self):
        """Test that DoRA computation matches mathematical definition."""
        layer = DoRAEmbedding(input_dim=3, output_dim=4, rank=2, alpha=1.0)
        layer.build(None)

        # Set known values for testing
        embeddings_val = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float32
        )
        lora_a_val = np.array(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32
        )
        lora_b_val = np.array(
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.float32
        )
        magnitude_val = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        layer.embeddings.assign(embeddings_val)
        layer.lora_a.assign(lora_a_val)
        layer.lora_b.assign(lora_b_val)
        layer.magnitude.assign(magnitude_val)

        # Manual computation
        lora_adaptation = np.matmul(lora_a_val, lora_b_val) * layer.scaling
        combined_embeddings = embeddings_val + lora_adaptation

        # Column-wise L2 norms
        column_norms = np.sqrt(
            np.sum(combined_embeddings**2, axis=0, keepdims=True)
        )
        normalized_embeddings = combined_embeddings / np.maximum(
            column_norms, 1e-8
        )
        expected_embeddings = normalized_embeddings * magnitude_val

        # Compare with layer output
        actual_embeddings = layer.get_effective_embeddings().numpy()
        np.testing.assert_allclose(
            actual_embeddings, expected_embeddings, rtol=1e-5
        )


class TestDoRAPositionEmbedding:
    """Test class for DoRAPositionEmbedding layer."""

    def setup_method(self):
        """Set up test fixtures."""
        keras.backend.clear_session()
        np.random.seed(42)
        tf.random.set_seed(42)

    def test_init(self):
        """Test DoRAPositionEmbedding initialization."""
        layer = DoRAPositionEmbedding(
            sequence_length=512, output_dim=128, rank=8, alpha=2.0
        )

        assert layer.sequence_length == 512
        assert layer.output_dim == 128
        assert layer.rank == 8
        assert layer.alpha == 2.0
        assert layer.scaling == 2.0 / 8

    def test_build(self):
        """Test layer building process."""
        layer = DoRAPositionEmbedding(
            sequence_length=100, output_dim=64, rank=4
        )
        layer.build((None, 10, 64))  # (batch_size, seq_len, hidden_dim)

        # Check weight shapes
        assert layer.position_embeddings.shape == (100, 64)
        assert layer.lora_a.shape == (100, 4)
        assert layer.lora_b.shape == (4, 64)
        assert layer.magnitude.shape == (64,)

        # Check trainability
        assert not layer.position_embeddings.trainable  # Frozen
        assert layer.lora_a.trainable
        assert layer.lora_b.trainable
        assert layer.magnitude.trainable

    def test_call_basic(self):
        """Test basic forward pass."""
        layer = DoRAPositionEmbedding(sequence_length=20, output_dim=16, rank=4)
        layer.build((None, 10, 16))

        # Input: token embeddings
        inputs = np.random.randn(2, 10, 16).astype(np.float32)

        outputs = layer(inputs)

        assert outputs.shape == (2, 10, 16)  # Same as input shape

    def test_call_with_start_index(self):
        """Test forward pass with custom start index."""
        layer = DoRAPositionEmbedding(sequence_length=50, output_dim=8, rank=2)
        layer.build((None, 5, 8))

        inputs = np.random.randn(3, 5, 8).astype(np.float32)

        # Test with different start indices
        outputs1 = layer(inputs, start_index=0)
        outputs2 = layer(inputs, start_index=10)

        assert outputs1.shape == outputs2.shape == (3, 5, 8)
        # Should produce different embeddings due to different positions
        assert not np.allclose(outputs1.numpy(), outputs2.numpy())

    def test_position_clipping(self):
        """Test that positions are properly clipped to valid range."""
        layer = DoRAPositionEmbedding(sequence_length=10, output_dim=4, rank=2)
        layer.build((None, 15, 4))  # seq_len > sequence_length

        inputs = np.random.randn(1, 15, 4).astype(np.float32)

        # Should not raise error even though seq_len > sequence_length
        outputs = layer(inputs)
        assert outputs.shape == (1, 15, 4)

    def test_get_config(self):
        """Test configuration serialization."""
        layer = DoRAPositionEmbedding(
            sequence_length=256, output_dim=512, rank=16, alpha=4.0
        )

        config = layer.get_config()

        assert config["sequence_length"] == 256
        assert config["output_dim"] == 512
        assert config["rank"] == 16
        assert config["alpha"] == 4.0


class TestConvertEmbeddingToDora:
    """Test class for Embedding to DoRA conversion utility."""

    def setup_method(self):
        """Set up test fixtures."""
        keras.backend.clear_session()
        np.random.seed(42)
        tf.random.set_seed(42)

    def test_convert_basic(self):
        """Test basic Embedding to DoRA conversion."""
        # Create and build original Embedding layer
        embedding = layers.Embedding(
            input_dim=100, output_dim=32, mask_zero=True
        )
        embedding.build(None)

        # Convert to DoRA
        dora = convert_embedding_to_dora(embedding, rank=8, alpha=2.0)

        # Check configuration transfer
        assert dora.input_dim == embedding.input_dim
        assert dora.output_dim == embedding.output_dim
        assert dora.mask_zero == embedding.mask_zero
        assert dora.rank == 8
        assert dora.alpha == 2.0

    def test_convert_preserves_weights(self):
        """Test that conversion preserves original weights."""
        # Create and build Embedding layer
        embedding = layers.Embedding(input_dim=50, output_dim=16)
        embedding.build(None)

        # Store original embeddings
        original_embeddings = embedding.embeddings.numpy().copy()

        # Convert to DoRA
        dora = convert_embedding_to_dora(embedding, rank=4)

        # Check that original embeddings are preserved in DoRA layer
        np.testing.assert_array_equal(
            dora.embeddings.numpy(), original_embeddings
        )

    def test_convert_unbuilt_layer(self):
        """Test converting unbuilt Embedding layer."""
        embedding = layers.Embedding(input_dim=200, output_dim=64)

        dora = convert_embedding_to_dora(embedding, rank=6)

        # Should work but layer shouldn't be built yet
        assert not dora.built
        assert dora.input_dim == 200
        assert dora.output_dim == 64

    def test_convert_functional_equivalence(self):
        """Test that converted layer produces same output initially."""
        # Create and build Embedding layer
        embedding = layers.Embedding(input_dim=20, output_dim=8)
        embedding.build(None)

        # Convert to DoRA
        dora = convert_embedding_to_dora(embedding)

        # Test with integer inputs
        inputs = np.array([[1, 5, 10, 3], [7, 2, 9, 4]], dtype=np.int32)

        embedding_output = embedding(inputs)
        dora_output = dora(inputs)

        # Should be approximately equal (small numerical differences expected)
        np.testing.assert_allclose(
            embedding_output.numpy(),
            dora_output.numpy(),
            rtol=1e-5,
            atol=1e-6,
            err_msg="DoRA output should match embeddings "
            "output after initialization",
        )
        """np.testing.assert_allclose(
            embedding_output.numpy(), dora_output.numpy(), rtol=1e-4
        )"""

    def test_convert_with_input_length(self):
        """Test converting Embedding layer with input_length specified."""
        embedding = layers.Embedding(
            input_dim=100, output_dim=32, input_length=10
        )

        dora = convert_embedding_to_dora(embedding)

        assert dora.input_dim == embedding.input_dim


class TestDoRAEmbeddingIntegration:
    """Integration tests for DoRA embedding layers."""

    def setup_method(self):
        """Set up test fixtures."""
        keras.backend.clear_session()
        np.random.seed(42)
        tf.random.set_seed(42)

    def test_in_transformer_model(self):
        """Test DoRA embeddings in a simple transformer-like model."""
        vocab_size = 1000
        seq_length = 32
        embed_dim = 128

        # Input
        inputs = layers.Input(shape=(seq_length,), dtype="int32")

        # Token embeddings with DoRA
        token_embeddings = DoRAEmbedding(
            input_dim=vocab_size, output_dim=embed_dim, rank=16, mask_zero=True
        )(inputs)

        # Position embeddings with DoRA
        position_embeddings = DoRAPositionEmbedding(
            sequence_length=seq_length, output_dim=embed_dim, rank=8
        )(token_embeddings)

        # Combine embeddings
        embeddings = layers.Add()([token_embeddings, position_embeddings])
        embeddings = layers.LayerNormalization()(embeddings)

        # Simple classifier head
        pooled = layers.GlobalAveragePooling1D()(embeddings)
        outputs = layers.Dense(2, activation="softmax")(pooled)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

        # Test with sample data
        x = np.random.randint(1, vocab_size, (16, seq_length))
        y = np.random.randint(0, 2, (16,))

        # Should train without errors
        history = model.fit(x, y, epochs=1, verbose=0)
        assert len(history.history["loss"]) == 1

    def test_save_and_load_with_custom_objects(self):
        """Test saving and loading models with DoRA embedding layers."""
        import os
        import tempfile

        # Create model with DoRA embeddings
        model = keras.Sequential(
            [
                DoRAEmbedding(input_dim=100, output_dim=32, rank=4),
                layers.GlobalAveragePooling1D(),
                layers.Dense(10, activation="softmax"),
            ]
        )

        # Generate test data and get predictions
        x = np.random.randint(0, 100, (8, 5))
        original_predictions = model.predict(x, verbose=0)

        # Save model
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.keras")
            model.save(model_path)

            # Load model with custom objects
            loaded_model = keras.models.load_model(
                model_path, custom_objects={"DoRAEmbedding": DoRAEmbedding}
            )

            # Test predictions are the same
            loaded_predictions = loaded_model.predict(x, verbose=0)
            np.testing.assert_allclose(
                original_predictions, loaded_predictions, rtol=1e-6
            )

    def test_gradient_flow_embeddings(self):
        """Test that gradients flow correctly through DoRA embedding layers."""
        model = keras.Sequential(
            [
                DoRAEmbedding(input_dim=50, output_dim=16, rank=4),
                layers.GlobalAveragePooling1D(),
                layers.Dense(1),
            ]
        )

        x = np.random.randint(0, 50, (4, 8))
        y = np.random.randn(4, 1).astype(np.float32)

        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = tf.reduce_mean(tf.square(predictions - y))

        # Get gradients
        gradients = tape.gradient(loss, model.trainable_variables)

        # Check that all trainable parameters have gradients
        # Check that all trainable parameters have gradients computed
        for grad in gradients:
            assert grad is not None

        # The gradients should have the correct shapes
        # Trainable vars in DoRAEmbedding:
        # - lora_a: (input_dim, rank) = (50, 4)
        # - lora_b: (rank, output_dim) = (4, 16)
        # - magnitude: (output_dim,) = (16,)
        # Plus Dense layer params:
        # - Dense kernel: (16, 1)
        # - Dense bias: (1,)
        expected_shapes = [
            (50, 4),  # lora_a
            (4, 16),  # lora_b
            (16,),  # magnitude
            (16, 1),  # Dense kernel
            (1,),  # Dense bias
        ]

        for grad, expected_shape in zip(gradients, expected_shapes):
            assert grad.shape == expected_shape

    def test_masking_propagation(self):
        """Test that masking propagates correctly through the model."""
        model = keras.Sequential(
            [
                DoRAEmbedding(
                    input_dim=20, output_dim=8, rank=2, mask_zero=True
                ),
                layers.LSTM(16, return_sequences=True),
                layers.Dense(1),
            ]
        )

        # Input with padding (zeros)
        x = np.array([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]], dtype=np.int32)

        # Should work without errors - masking should handle padding
        outputs = model(x)
        assert outputs.shape == (2, 5, 1)

    def test_vocabulary_expansion_in_model(self):
        """Test vocabulary expansion with a model."""
        # Create initial model
        embedding_layer = DoRAEmbedding(input_dim=10, output_dim=8, rank=2)
        model = keras.Sequential(
            [
                embedding_layer,
                layers.GlobalAveragePooling1D(),
                layers.Dense(2, activation="softmax"),
            ]
        )

        # Build model
        model.build((None, 5))

        # Train on initial vocabulary
        x = np.random.randint(0, 10, (16, 5))
        y = np.random.randint(0, 2, (16,))
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        model.fit(x, y, epochs=1, verbose=0)

        # Expand vocabulary
        expanded_embedding = embedding_layer.expand_vocabulary(15)

        # Create new model with expanded vocabulary
        new_model = keras.Sequential(
            [
                expanded_embedding,
                layers.GlobalAveragePooling1D(),
                layers.Dense(2, activation="softmax"),
            ]
        )

        # Test with expanded vocabulary
        x_expanded = np.random.randint(
            0, 15, (8, 5)
        )  # Can now use tokens 10-14
        new_model.compile(
            optimizer="adam", loss="sparse_categorical_crossentropy"
        )

        # Should work without errors
        predictions = new_model.predict(x_expanded, verbose=0)
        assert predictions.shape == (8, 2)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        keras.backend.clear_session()
        np.random.seed(42)
        tf.random.set_seed(42)

    def test_very_small_embeddings(self):
        """Test with very small embedding dimensions."""
        layer = DoRAEmbedding(input_dim=2, output_dim=1, rank=1)
        layer.build(None)

        inputs = np.array([[0], [1]], dtype=np.int32)
        outputs = layer(inputs)

        assert outputs.shape == (2, 1, 1)

    def test_rank_larger_than_dimensions(self):
        """Test with rank larger than input/output dimensions."""
        # This should work but be inefficient
        layer = DoRAEmbedding(input_dim=5, output_dim=3, rank=10)
        layer.build(None)

        inputs = np.array([[0, 1, 2]], dtype=np.int32)
        outputs = layer(inputs)

        assert outputs.shape == (1, 3, 3)

    def test_zero_magnitude_initialization(self):
        """Test behavior with zero magnitude initialization."""
        layer = DoRAEmbedding(
            input_dim=5, output_dim=3, rank=2, magnitude_initializer="zeros"
        )
        layer.build(None)

        inputs = np.array([[0, 1, 2]], dtype=np.int32)
        outputs = layer(inputs)

        # Output should be close to zero due to zero magnitudes
        assert np.allclose(outputs.numpy(), 0, atol=1e-6)

    def test_very_large_alpha(self):
        """Test with very large alpha value."""
        layer = DoRAEmbedding(input_dim=5, output_dim=3, rank=2, alpha=1000.0)
        layer.build(None)

        inputs = np.array([[0, 1]], dtype=np.int32)
        outputs = layer(inputs)

        # Should not cause numerical issues
        assert not np.any(np.isnan(outputs.numpy()))
        assert not np.any(np.isinf(outputs.numpy()))


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
