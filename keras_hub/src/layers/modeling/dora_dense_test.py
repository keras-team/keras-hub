"""Test suite for DoRA Dense Layer Implementation.

This test suite is backend-independent and works with TensorFlow,
PyTorch, and JAX.
Run with: python -m pytest test_dora_dense.py -v
"""

import keras
import numpy as np
import pytest
from keras import layers
from keras import ops

from .dora_dense import DoRADense
from .dora_dense import convert_dense_to_dora


class TestDoRADense:
    """Test cases for DoRADense layer."""

    @pytest.fixture
    def sample_input(self):
        """Create sample input data."""
        return np.random.randn(32, 64).astype(np.float32)

    @pytest.fixture
    def dora_layer(self):
        """Create a basic DoRA layer."""
        return DoRADense(
            units=128, rank=8, alpha=2.0, use_bias=True, activation="relu"
        )

    def test_layer_creation(self):
        """Test basic layer creation with various configurations."""
        # Test default parameters
        layer = DoRADense(units=64)
        assert layer.units == 64
        assert layer.rank == 4
        assert layer.alpha == 1.0
        assert layer.use_bias is True
        assert layer.dropout_rate == 0.0

        # Test custom parameters
        layer = DoRADense(
            units=128,
            rank=16,
            alpha=0.5,
            use_bias=False,
            dropout=0.2,
            activation="tanh",
        )
        assert layer.units == 128
        assert layer.rank == 16
        assert layer.alpha == 0.5
        assert layer.use_bias is False
        assert layer.dropout_rate == 0.2
        assert layer.activation == keras.activations.tanh

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test invalid units
        with pytest.raises(ValueError, match="units must be positive"):
            DoRADense(units=0)

        with pytest.raises(ValueError, match="units must be positive"):
            DoRADense(units=-5)

        # Test invalid rank
        with pytest.raises(ValueError, match="rank must be positive"):
            DoRADense(units=64, rank=0)

        with pytest.raises(ValueError, match="rank must be positive"):
            DoRADense(units=64, rank=-2)

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

    def test_layer_build(self, sample_input):
        """Test layer building process."""
        layer = DoRADense(units=32, rank=4)

        # Layer should not be built initially
        assert not layer.built

        # Build the layer
        layer.build(sample_input.shape)

        # Check if layer is built
        assert layer.built

        # Check weight shapes
        input_dim = sample_input.shape[-1]
        assert layer.kernel.shape == (input_dim, 32)
        assert layer.lora_a.shape == (input_dim, 4)
        assert layer.lora_b.shape == (4, 32)
        assert layer.magnitude.shape == (32,)
        assert layer.bias.shape == (32,)

    def test_forward_pass(self, sample_input, dora_layer):
        """Test forward pass functionality."""
        # Build and run forward pass
        output = dora_layer(sample_input)

        # Check output shape
        expected_shape = (sample_input.shape[0], dora_layer.units)
        assert output.shape == expected_shape

        # Check output is not NaN or Inf
        output_np = ops.convert_to_numpy(output)
        assert not np.isnan(output_np).any()
        assert not np.isinf(output_np).any()

    def test_weight_initialization(self, sample_input):
        """Test weight initialization."""
        layer = DoRADense(
            units=32,
            rank=4,
            lora_a_initializer="he_uniform",
            lora_b_initializer="zeros",
            magnitude_initializer="ones",
        )

        # Build the layer
        layer.build(sample_input.shape)

        # Check lora_b is initialized to zeros
        lora_b_np = ops.convert_to_numpy(layer.lora_b)
        assert np.allclose(lora_b_np, 0.0)

        # Check magnitude is initialized to ones
        magnitude_np = ops.convert_to_numpy(layer.magnitude)
        assert np.allclose(magnitude_np, 1.0)

    def test_activation_functions(self, sample_input):
        """Test different activation functions."""
        activations = ["relu", "tanh", "sigmoid", "linear", None]

        for activation in activations:
            layer = DoRADense(units=16, activation=activation)
            output = layer(sample_input)

            # Check output shape
            assert output.shape == (sample_input.shape[0], 16)

            # Check activation is applied correctly
            if activation == "relu":
                output_np = ops.convert_to_numpy(output)
                assert (output_np >= 0).all()

    def test_bias_configuration(self, sample_input):
        """Test bias configuration."""
        # With bias
        layer_with_bias = DoRADense(units=16, use_bias=True)
        layer_with_bias.build(sample_input.shape)
        assert layer_with_bias.bias is not None

        # Without bias
        layer_without_bias = DoRADense(units=16, use_bias=False)
        layer_without_bias.build(sample_input.shape)
        assert layer_without_bias.bias is None

    def test_dropout_functionality(self, sample_input):
        """Test dropout functionality."""
        layer_no_dropout = DoRADense(units=16, dropout=0.0)
        layer_with_dropout = DoRADense(units=16, dropout=0.5)

        # Test without dropout
        output_no_dropout = layer_no_dropout(sample_input, training=True)
        assert output_no_dropout.shape == (sample_input.shape[0], 16)

        # Test with dropout
        output_with_dropout = layer_with_dropout(sample_input, training=True)
        assert output_with_dropout.shape == (sample_input.shape[0], 16)

    def test_get_effective_weight(self, sample_input, dora_layer):
        """Test effective weight computation."""
        # Build the layer first
        dora_layer.build(sample_input.shape)

        # Get effective weight
        effective_weight = dora_layer.get_effective_weight()

        # Check shape
        input_dim = sample_input.shape[-1]
        expected_shape = (input_dim, dora_layer.units)
        assert effective_weight.shape == expected_shape

        # Check it's not NaN or Inf
        weight_np = ops.convert_to_numpy(effective_weight)
        assert not np.isnan(weight_np).any()
        assert not np.isinf(weight_np).any()

    def test_get_dora_parameters(self, sample_input, dora_layer):
        """Test DoRA parameter retrieval."""
        dora_layer.build(sample_input.shape)

        params = dora_layer.get_dora_parameters()

        # Check all expected parameters are present
        assert "lora_a" in params
        assert "lora_b" in params
        assert "magnitude" in params
        assert "bias" in params  # Since use_bias=True by default

        # Check shapes
        input_dim = sample_input.shape[-1]
        assert params["lora_a"].shape == (input_dim, dora_layer.rank)
        assert params["lora_b"].shape == (dora_layer.rank, dora_layer.units)
        assert params["magnitude"].shape == (dora_layer.units,)
        assert params["bias"].shape == (dora_layer.units,)

    def test_merge_weights(self, sample_input, dora_layer):
        """Test weight merging functionality."""
        dora_layer.build(sample_input.shape)

        merged = dora_layer.merge_weights()

        # Check structure
        assert "kernel" in merged
        assert "bias" in merged

        # Check shapes
        input_dim = sample_input.shape[-1]
        assert merged["kernel"].shape == (input_dim, dora_layer.units)
        assert merged["bias"].shape == (dora_layer.units,)

    def test_count_params(self, sample_input):
        """Test parameter counting."""
        layer = DoRADense(units=32, rank=8, use_bias=True)

        # Should return 0 before building
        assert layer.count_params() == 0

        # Build and count
        layer.build(sample_input.shape)
        input_dim = sample_input.shape[-1]

        expected_params = (
            input_dim * 8  # lora_a
            + 8 * 32  # lora_b
            + 32  # magnitude
            + 32  # bias
        )

        assert layer.count_params() == expected_params

    def test_load_pretrained_weights(self, sample_input):
        """Test loading pretrained weights."""
        layer = DoRADense(units=32, rank=4)
        layer.build(sample_input.shape)

        input_dim = sample_input.shape[-1]

        # Create fake pretrained weights
        pretrained_kernel = np.random.randn(input_dim, 32).astype(np.float32)
        pretrained_bias = np.random.randn(32).astype(np.float32)

        # Load weights
        layer.load_pretrained_weights(pretrained_kernel, pretrained_bias)

        # Check if weights are loaded correctly
        kernel_np = ops.convert_to_numpy(layer.kernel)
        bias_np = ops.convert_to_numpy(layer.bias)

        assert np.allclose(kernel_np, pretrained_kernel)
        assert np.allclose(bias_np, pretrained_bias)

        # Check magnitude is initialized to column norms
        expected_magnitude = np.linalg.norm(pretrained_kernel, axis=0)
        magnitude_np = ops.convert_to_numpy(layer.magnitude)
        assert np.allclose(magnitude_np, expected_magnitude, rtol=1e-5)

    def test_load_pretrained_weights_shape_mismatch(self, sample_input):
        """Test loading pretrained weights with wrong shapes."""
        layer = DoRADense(units=32, rank=4)
        layer.build(sample_input.shape)

        # Wrong kernel shape
        wrong_kernel = np.random.randn(10, 20).astype(np.float32)
        with pytest.raises(ValueError, match="Pretrained kernel shape"):
            layer.load_pretrained_weights(wrong_kernel)

        # Wrong bias shape
        correct_kernel = np.random.randn(sample_input.shape[-1], 32).astype(
            np.float32
        )
        wrong_bias = np.random.randn(20).astype(np.float32)
        with pytest.raises(ValueError, match="Pretrained bias shape"):
            layer.load_pretrained_weights(correct_kernel, wrong_bias)

    def test_serialization(self, dora_layer):
        """Test layer serialization and deserialization."""
        # Get config
        config = dora_layer.get_config()

        # Check essential parameters are in config
        assert config["units"] == dora_layer.units
        assert config["rank"] == dora_layer.rank
        assert config["alpha"] == dora_layer.alpha
        assert config["use_bias"] == dora_layer.use_bias
        assert config["dropout"] == dora_layer.dropout_rate

        # Create layer from config
        restored_layer = DoRADense.from_config(config)

        # Check restored layer has same parameters
        assert restored_layer.units == dora_layer.units
        assert restored_layer.rank == dora_layer.rank
        assert restored_layer.alpha == dora_layer.alpha
        assert restored_layer.use_bias == dora_layer.use_bias

    def test_compute_output_shape(self):
        """Test output shape computation."""
        layer = DoRADense(units=64)

        # Test various input shapes
        input_shapes = [
            (None, 32),
            (10, 32),
            (None, 16, 32),
            (5, 10, 32),
        ]

        for input_shape in input_shapes:
            output_shape = layer.compute_output_shape(input_shape)
            expected_shape = input_shape[:-1] + (64,)
            assert output_shape == expected_shape

    def test_regularization(self, sample_input):
        """Test regularization functionality."""
        layer = DoRADense(
            units=32,
            kernel_regularizer="l2",
            bias_regularizer="l1",
            activity_regularizer="l2",
        )

        # Build and run forward pass
        output = layer(sample_input)

        # Check output shape
        assert output.shape == (sample_input.shape[0], 32)

    def test_constraints(self, sample_input):
        """Test constraint functionality."""
        layer = DoRADense(
            units=32, kernel_constraint="max_norm", bias_constraint="non_neg"
        )

        # Build and run forward pass
        output = layer(sample_input)

        # Check output shape
        assert output.shape == (sample_input.shape[0], 32)

    def test_training_inference_consistency(self, sample_input, dora_layer):
        """Test consistency between training and inference modes."""
        # Forward pass in training mode
        output_train = dora_layer(sample_input, training=True)

        # Forward pass in inference mode
        output_infer = dora_layer(sample_input, training=False)

        # Should have same shape
        assert output_train.shape == output_infer.shape

        # For layers without dropout, outputs should be identical
        if dora_layer.dropout_rate == 0:
            output_train_np = ops.convert_to_numpy(output_train)
            output_infer_np = ops.convert_to_numpy(output_infer)
            assert np.allclose(output_train_np, output_infer_np)


class TestDoRAConversion:
    """Test cases for Dense to DoRA conversion."""

    def test_convert_dense_to_dora(self):
        """Test converting Dense layer to DoRA layer."""
        # Create a Dense layer
        dense_layer = layers.Dense(
            units=64,
            activation="relu",
            use_bias=True,
            kernel_initializer="glorot_uniform",
        )

        # Build with sample input
        sample_input = np.random.randn(10, 32).astype(np.float32)
        dense_output = dense_layer(sample_input)

        # Convert to DoRA
        dora_layer = convert_dense_to_dora(
            dense_layer, rank=8, alpha=2.0, dropout=0.1
        )

        # Check configuration
        assert dora_layer.units == dense_layer.units
        assert dora_layer.rank == 8
        assert dora_layer.alpha == 2.0
        assert dora_layer.dropout_rate == 0.1
        assert dora_layer.use_bias == dense_layer.use_bias
        assert dora_layer.activation == dense_layer.activation

        # Check weights are loaded
        assert dora_layer.built

        # Test forward pass produces reasonable output
        dora_output = dora_layer(sample_input)
        assert dora_output.shape == dense_output.shape

    def test_convert_unbuilt_dense(self):
        """Test converting unbuilt Dense layer."""
        dense_layer = layers.Dense(units=32, activation="tanh")

        # Convert unbuilt layer
        dora_layer = convert_dense_to_dora(dense_layer, rank=4)

        # Should not be built yet
        assert not dora_layer.built

        # But should have correct configuration
        assert dora_layer.units == 32
        assert dora_layer.rank == 4
        assert dora_layer.activation == keras.activations.tanh


class TestDoRAMathematicalProperties:
    """Test mathematical properties of DoRA."""

    def test_magnitude_scaling_property(self):
        """Test that DoRA properly applies magnitude scaling."""
        # Create layer
        layer = DoRADense(units=16, rank=4)
        sample_input = np.random.randn(8, 32).astype(np.float32)
        layer.build(sample_input.shape)

        # Get effective weight
        effective_weight = layer.get_effective_weight()
        effective_weight_np = ops.convert_to_numpy(effective_weight)

        # Compute column norms of effective weight
        column_norms = np.linalg.norm(effective_weight_np, axis=0)
        magnitude_np = ops.convert_to_numpy(layer.magnitude)

        # Column norms should equal magnitude values (approximately)
        assert np.allclose(column_norms, magnitude_np, rtol=1e-5)

    def test_low_rank_adaptation_property(self):
        """Test that adaptation is indeed low-rank."""
        layer = DoRADense(units=64, rank=8)
        sample_input = np.random.randn(16, 128).astype(np.float32)
        layer.build(sample_input.shape)

        # Compute LoRA adaptation
        lora_a_np = ops.convert_to_numpy(layer.lora_a)
        lora_b_np = ops.convert_to_numpy(layer.lora_b)
        adaptation = lora_a_np @ lora_b_np

        # Check that adaptation matrix has rank <= layer.rank
        actual_rank = np.linalg.matrix_rank(adaptation)
        assert actual_rank <= layer.rank

    def test_zero_initialization_equivalence(self):
        """Test that zero LoRA initialization gives original behavior."""
        # Create layer with zero LoRA initialization
        layer = DoRADense(
            units=32,
            rank=4,
            lora_a_initializer="zeros",
            lora_b_initializer="zeros",
        )

        sample_input = np.random.randn(8, 16).astype(np.float32)
        layer.build(sample_input.shape)

        # Set magnitude to column norms of kernel
        kernel_np = ops.convert_to_numpy(layer.kernel)
        column_norms = np.linalg.norm(kernel_np, axis=0)
        layer.magnitude.assign(column_norms)

        # Effective weight should equal original kernel
        effective_weight = layer.get_effective_weight()
        effective_weight_np = ops.convert_to_numpy(effective_weight)

        assert np.allclose(effective_weight_np, kernel_np, rtol=1e-5)


def test_backend_compatibility():
    """Test that the implementation works across different backends."""
    # This test ensures the code runs without backend-specific errors
    layer = DoRADense(units=16, rank=4)
    sample_input = np.random.randn(4, 8).astype(np.float32)

    # Should work regardless of backend
    output = layer(sample_input)
    assert output.shape == (4, 16)

    # Test parameter access
    params = layer.get_dora_parameters()
    assert len(params) == 4  # lora_a, lora_b, magnitude, bias

    backend = keras.backend.backend()
    assert backend in ["tensorflow", "torch", "jax"], (
        f"Backend compatibility test failed. "
        f"Expected one of ['tensorflow', 'torch', 'jax'], got: {backend}"
    )
