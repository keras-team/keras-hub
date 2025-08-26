"""Test suite for DoRA Dense Layer Implementation.

This module contains comprehensive tests for the DoRADense layer,
including functionality, compatibility, and edge cases.
"""

import keras
import numpy as np
import pytest
import tensorflow as tf
from keras import layers

# Import the module to test
from .dora_dense import DoRADense
from .dora_dense import convert_dense_to_dora


class TestDoRADense:
    """Test class for DoRADense layer."""

    def setup_method(self):
        """Set up test fixtures."""
        # Clear any existing session
        keras.backend.clear_session()

        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)

    def test_init_valid_params(self):
        """Test DoRADense initialization with valid parameters."""
        layer = DoRADense(
            units=64,
            rank=8,
            alpha=2.0,
            use_bias=True,
            dropout=0.1,
            activation="relu",
        )

        assert layer.units == 64
        assert layer.rank == 8
        assert layer.alpha == 2.0
        assert layer.use_bias is True
        assert layer.dropout_rate == 0.1
        assert layer.scaling == 2.0 / 8  # alpha / rank

    def test_init_invalid_params(self):
        """Test DoRADense initialization with invalid parameters."""
        # Test invalid units
        with pytest.raises(ValueError, match="units must be positive"):
            DoRADense(units=0)

        with pytest.raises(ValueError, match="units must be positive"):
            DoRADense(units=-10)

        # Test invalid rank
        with pytest.raises(ValueError, match="rank must be positive"):
            DoRADense(units=64, rank=0)

        with pytest.raises(ValueError, match="rank must be positive"):
            DoRADense(units=64, rank=-5)

        # Test invalid alpha
        with pytest.raises(ValueError, match="alpha must be positive"):
            DoRADense(units=64, alpha=0)

        with pytest.raises(ValueError, match="alpha must be positive"):
            DoRADense(units=64, alpha=-1.0)

        # Test invalid dropout
        with pytest.raises(ValueError, match="dropout must be in"):
            DoRADense(units=64, dropout=1.0)

        with pytest.raises(ValueError, match="dropout must be in"):
            DoRADense(units=64, dropout=-0.1)

    def test_build(self):
        """Test layer building process."""
        layer = DoRADense(units=32, rank=4)
        input_shape = (None, 16)

        layer.build(input_shape)

        # Check that weights are created
        assert layer.kernel is not None
        assert layer.lora_a is not None
        assert layer.lora_b is not None
        assert layer.magnitude is not None
        assert layer.bias is not None

        # Check weight shapes
        assert layer.kernel.shape == (16, 32)
        assert layer.lora_a.shape == (16, 4)
        assert layer.lora_b.shape == (4, 32)
        assert layer.magnitude.shape == (32,)
        assert layer.bias.shape == (32,)

        # Check trainability
        assert not layer.kernel.trainable  # Frozen
        assert layer.lora_a.trainable
        assert layer.lora_b.trainable
        assert layer.magnitude.trainable
        assert layer.bias.trainable

    def test_build_no_bias(self):
        """Test layer building without bias."""
        layer = DoRADense(units=32, rank=4, use_bias=False)
        input_shape = (None, 16)

        layer.build(input_shape)

        assert layer.bias is None

    def test_build_invalid_input_shape(self):
        """Test building with invalid input shapes."""
        layer = DoRADense(units=32)

        # Test with insufficient dimensions
        with pytest.raises(ValueError, match="must have at least 2 dimensions"):
            layer.build((10,))

        # Test with undefined last dimension
        with pytest.raises(ValueError, match="last dimension.*must be defined"):
            layer.build((None, None))

    def test_call_basic(self):
        """Test basic forward pass."""
        layer = DoRADense(units=8, rank=2, activation="relu")
        inputs = np.random.randn(4, 16).astype(np.float32)

        layer.build((None, 16))
        outputs = layer(inputs)

        assert outputs.shape == (4, 8)
        assert np.all(outputs.numpy() >= 0)  # ReLU activation

    def test_call_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        layer = DoRADense(units=10, rank=4)
        layer.build((None, 5))

        # Test different batch sizes
        for batch_size in [1, 8, 32]:
            inputs = np.random.randn(batch_size, 5).astype(np.float32)
            outputs = layer(inputs)
            assert outputs.shape == (batch_size, 10)

    def test_call_with_dropout(self):
        """Test forward pass with dropout."""
        layer = DoRADense(units=16, rank=4, dropout=0.5)
        inputs = np.random.randn(8, 12).astype(np.float32)

        layer.build((None, 12))

        # Training mode (dropout active)
        outputs_train = layer(inputs, training=True)

        # Inference mode (no dropout)
        outputs_inf = layer(inputs, training=False)

        assert outputs_train.shape == outputs_inf.shape == (8, 16)

    def test_get_dora_parameters(self):
        """Test getting DoRA parameters."""
        layer = DoRADense(units=16, rank=4)
        layer.build((None, 8))

        params = layer.get_dora_parameters()

        assert "lora_a" in params
        assert "lora_b" in params
        assert "magnitude" in params
        assert "bias" in params

        assert params["lora_a"] is layer.lora_a
        assert params["lora_b"] is layer.lora_b
        assert params["magnitude"] is layer.magnitude
        assert params["bias"] is layer.bias

    def test_get_dora_parameters_no_bias(self):
        """Test getting DoRA parameters without bias."""
        layer = DoRADense(units=16, rank=4, use_bias=False)
        layer.build((None, 8))

        params = layer.get_dora_parameters()

        assert "bias" not in params

    def test_get_effective_weight(self):
        """Test computing effective weight matrix."""
        layer = DoRADense(units=8, rank=2)
        layer.build((None, 4))

        effective_weight = layer.get_effective_weight()

        assert effective_weight.shape == (4, 8)

        # Test that it's different from original kernel
        assert not np.allclose(effective_weight.numpy(), layer.kernel.numpy())

    def test_merge_weights(self):
        """Test merging DoRA weights."""
        layer = DoRADense(units=6, rank=2)
        layer.build((None, 3))

        merged = layer.merge_weights()

        assert "kernel" in merged
        assert "bias" in merged
        assert merged["kernel"].shape == (3, 6)
        assert merged["bias"].shape == (6,)

    def test_merge_weights_no_bias(self):
        """Test merging weights without bias."""
        layer = DoRADense(units=6, rank=2, use_bias=False)
        layer.build((None, 3))

        merged = layer.merge_weights()

        assert "kernel" in merged
        assert "bias" not in merged

    def test_count_params(self):
        """Test parameter counting."""
        # Test with bias
        layer = DoRADense(units=10, rank=4, use_bias=True)
        layer.build((None, 8))

        expected_params = (
            8 * 4  # lora_a: input_dim * rank
            + 4 * 10  # lora_b: rank * units
            + 10  # magnitude: units
            + 10  # bias: units
        )
        assert layer.count_params() == expected_params

        # Test without bias
        layer_no_bias = DoRADense(units=10, rank=4, use_bias=False)
        layer_no_bias.build((None, 8))

        expected_params_no_bias = 8 * 4 + 4 * 10 + 10
        assert layer_no_bias.count_params() == expected_params_no_bias

    def test_count_params_unbuilt(self):
        """Test parameter counting for unbuilt layer."""
        layer = DoRADense(units=10, rank=4)
        assert layer.count_params() == 0

    def test_load_pretrained_weights(self):
        """Test loading pretrained weights."""
        layer = DoRADense(units=6, rank=2)
        layer.build((None, 4))

        # Create pretrained weights
        pretrained_kernel = np.random.randn(4, 6).astype(np.float32)
        pretrained_bias = np.random.randn(6).astype(np.float32)

        # Store original values
        original_kernel = layer.kernel.numpy().copy()
        original_bias = layer.bias.numpy().copy()

        # Load pretrained weights
        layer.load_pretrained_weights(pretrained_kernel, pretrained_bias)

        # Check that weights changed
        np.testing.assert_array_equal(layer.kernel.numpy(), pretrained_kernel)
        np.testing.assert_array_equal(layer.bias.numpy(), pretrained_bias)
        assert not np.allclose(layer.kernel.numpy(), original_kernel)
        assert not np.allclose(layer.bias.numpy(), original_bias)

    def test_load_pretrained_weights_shape_mismatch(self):
        """Test loading pretrained weights with wrong shapes."""
        layer = DoRADense(units=6, rank=2)
        layer.build((None, 4))

        # Wrong kernel shape
        wrong_kernel = np.random.randn(5, 6).astype(np.float32)
        with pytest.raises(ValueError, match="doesn't match expected shape"):
            layer.load_pretrained_weights(wrong_kernel)

        # Wrong bias shape
        correct_kernel = np.random.randn(4, 6).astype(np.float32)
        wrong_bias = np.random.randn(5).astype(np.float32)
        with pytest.raises(ValueError, match="doesn't match expected shape"):
            layer.load_pretrained_weights(correct_kernel, wrong_bias)

    def test_get_config(self):
        """Test layer configuration serialization."""
        layer = DoRADense(
            units=32,
            rank=8,
            alpha=2.0,
            use_bias=False,
            dropout=0.2,
            activation="tanh",
        )

        config = layer.get_config()

        assert config["units"] == 32
        assert config["rank"] == 8
        assert config["alpha"] == 2.0
        assert config["use_bias"] is False
        assert config["dropout"] == 0.2

    def test_from_config(self):
        """Test layer creation from configuration."""
        original_layer = DoRADense(units=16, rank=4, alpha=1.5)
        config = original_layer.get_config()

        new_layer = DoRADense.from_config(config)

        assert new_layer.units == original_layer.units
        assert new_layer.rank == original_layer.rank
        assert new_layer.alpha == original_layer.alpha

    def test_compute_output_shape(self):
        """Test output shape computation."""
        layer = DoRADense(units=20)

        output_shape = layer.compute_output_shape((None, 10))
        assert output_shape == (None, 20)

        output_shape = layer.compute_output_shape((32, 15))
        assert output_shape == (32, 20)

        output_shape = layer.compute_output_shape((4, 8, 10))
        assert output_shape == (4, 8, 20)

    def test_mathematical_correctness(self):
        """Test that DoRA computation matches mathematical definition."""
        layer = DoRADense(
            units=4, rank=2, alpha=1.0, use_bias=False, activation=None
        )
        layer.build((None, 3))

        # Set known values for testing
        kernel_val = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float32
        )
        lora_a_val = np.array(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32
        )
        lora_b_val = np.array(
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.float32
        )
        magnitude_val = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        layer.kernel.assign(kernel_val)
        layer.lora_a.assign(lora_a_val)
        layer.lora_b.assign(lora_b_val)
        layer.magnitude.assign(magnitude_val)

        # Manual computation
        lora_adaptation = np.matmul(lora_a_val, lora_b_val) * layer.scaling
        combined_weight = kernel_val + lora_adaptation

        # Column-wise L2 norms
        column_norms = np.sqrt(
            np.sum(combined_weight**2, axis=0, keepdims=True)
        )
        normalized_weight = combined_weight / np.maximum(column_norms, 1e-8)
        expected_weight = normalized_weight * magnitude_val

        # Compare with layer output
        actual_weight = layer.get_effective_weight().numpy()
        np.testing.assert_allclose(actual_weight, expected_weight, rtol=1e-5)


class TestConvertDenseToDora:
    """Test class for Dense to DoRA conversion utility."""

    def setup_method(self):
        """Set up test fixtures."""
        keras.backend.clear_session()
        np.random.seed(42)
        tf.random.set_seed(42)

    def test_convert_basic(self):
        """Test basic Dense to DoRA conversion."""
        # Create and build original Dense layer
        dense = layers.Dense(units=16, activation="relu", use_bias=True)
        dense.build((None, 8))

        # Convert to DoRA
        dora = convert_dense_to_dora(dense, rank=4, alpha=2.0)

        # Check configuration transfer
        assert dora.units == dense.units
        assert dora.activation == dense.activation
        assert dora.use_bias == dense.use_bias
        assert dora.rank == 4
        assert dora.alpha == 2.0

    def test_convert_preserves_weights(self):
        """Test that conversion preserves original weights."""
        # Create, build, and initialize Dense layer
        dense = layers.Dense(units=10, use_bias=True)
        dense.build((None, 5))

        # Store original weights
        original_kernel = dense.kernel.numpy().copy()
        original_bias = dense.bias.numpy().copy()

        # Convert to DoRA
        dora = convert_dense_to_dora(dense, rank=2)

        # Check that original weights are preserved in DoRA layer
        np.testing.assert_array_equal(dora.kernel.numpy(), original_kernel)
        np.testing.assert_array_equal(dora.bias.numpy(), original_bias)

    def test_convert_unbuilt_layer(self):
        """Test converting unbuilt Dense layer."""
        dense = layers.Dense(units=12, activation="tanh")

        dora = convert_dense_to_dora(dense, rank=3)

        # Should work but layer shouldn't be built yet
        assert not dora.built
        assert dora.units == 12

    def test_convert_functional_equivalence(self):
        """Test that converted DoRA layer
        preserves output initially."""
        # Create and build Dense layer
        dense = layers.Dense(units=8, use_bias=True, activation=None)
        dense.build((None, 4))

        # Convert to DoRA
        dora = convert_dense_to_dora(dense)

        # Test input
        inputs = np.random.randn(2, 4).astype(np.float32)

        dense_output = dense(inputs)
        dora_output = dora(inputs)

        # Check that outputs have the same shape
        assert dense_output.shape == dora_output.shape

        # After proper initialization,
        # DoRA should behave identically to Dense
        # Allow for small numerical differences
        # due to floating point precision
        np.testing.assert_allclose(
            dense_output.numpy(),
            dora_output.numpy(),
            rtol=1e-5,
            atol=1e-6,
            err_msg="DoRA output should match "
            "Dense output after initialization",
        )

    def test_magnitude_initialization(self):
        """Test that magnitude vector is properly
        initialized to column norms."""
        # Create and build Dense layer
        dense = layers.Dense(units=6, use_bias=False, activation=None)
        dense.build((None, 4))

        # Store original kernel
        original_kernel = dense.kernel.numpy()

        # Convert to DoRA
        dora = convert_dense_to_dora(dense)

        # Calculate expected magnitude (column-wise norms)
        expected_magnitude = np.sqrt(np.sum(original_kernel**2, axis=0))

        # Check that magnitude was initialized correctly
        np.testing.assert_allclose(
            dora.magnitude.numpy(),
            expected_magnitude,
            rtol=1e-6,
            err_msg="Magnitude should be initialized to "
            "column-wise norms of pretrained weights",
        )


class TestDoRADenseIntegration:
    """Integration tests for DoRADense layer."""

    def setup_method(self):
        """Set up test fixtures."""
        keras.backend.clear_session()
        np.random.seed(42)
        tf.random.set_seed(42)

    def test_in_sequential_model(self):
        """Test DoRADense in a Sequential model."""
        model = keras.Sequential(
            [
                layers.Input(shape=(10,)),
                DoRADense(units=16, rank=4, activation="relu"),
                DoRADense(units=8, rank=2, activation="relu"),
                DoRADense(units=1, rank=1, activation="sigmoid"),
            ]
        )

        model.compile(optimizer="adam", loss="binary_crossentropy")

        # Test with sample data
        x = np.random.randn(32, 10).astype(np.float32)
        y = np.random.randint(0, 2, (32, 1)).astype(np.float32)

        # Should train without errors
        history = model.fit(x, y, epochs=2, verbose=0)
        assert len(history.history["loss"]) == 2

    def test_in_functional_model(self):
        """Test DoRADense in a Functional model."""
        inputs = layers.Input(shape=(15,))
        x = DoRADense(units=20, rank=4, activation="relu")(inputs)
        x = layers.Dropout(0.2)(x)
        outputs = DoRADense(units=5, rank=2, activation="softmax")(x)

        model = keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

        # Test with sample data
        x = np.random.randn(16, 15).astype(np.float32)
        y = np.random.randint(0, 5, (16,))

        # Should train without errors
        model.fit(x, y, epochs=1, verbose=0)

    def test_save_and_load(self):
        """Test saving and loading models with DoRADense layers."""
        import os
        import tempfile

        # Create model
        model = keras.Sequential(
            [
                layers.Input(shape=(6,)),
                DoRADense(units=4, rank=2, activation="relu"),
                DoRADense(units=2, rank=1),
            ]
        )

        # Generate test data and get predictions
        x = np.random.randn(8, 6).astype(np.float32)
        original_predictions = model.predict(x, verbose=0)

        # Save model
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.keras")
            model.save(model_path)

            # Load model
            loaded_model = keras.models.load_model(
                model_path, custom_objects={"DoRADense": DoRADense}
            )

            # Test predictions are the same
            loaded_predictions = loaded_model.predict(x, verbose=0)
            np.testing.assert_allclose(
                original_predictions, loaded_predictions, rtol=1e-6
            )

    def test_gradient_flow(self):
        """Test that gradients flow correctly through DoRADense."""
        model = keras.Sequential(
            [layers.Input(shape=(4,)), DoRADense(units=3, rank=2)]
        )

        x = np.random.randn(2, 4).astype(np.float32)
        y = np.random.randn(2, 3).astype(np.float32)

        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = tf.reduce_mean(tf.square(predictions - y))

        # Get gradients
        gradients = tape.gradient(loss, model.trainable_variables)

        # Check that all trainable parameters have gradients computed
        for grad in gradients:
            assert grad is not None

        # The gradients should have the correct shapes and types
        # Note: lora_a gradient might be zero initially
        # due to lora_b being zero-initialized
        # This is mathematically correct behavior, not an error
        expected_shapes = [
            (4, 2),
            (2, 3),
            (3,),
            (3,),
        ]  # lora_a, lora_b, magnitude, bias
        for grad, expected_shape in zip(gradients, expected_shapes):
            assert grad.shape == expected_shape


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
