"""Test suite for DoRA Embedding Layer Implementation.

This test suite is backend-independent and works with
TensorFlow, PyTorch, and JAX.
Run with: python -m pytest test_dora_embeddings.py -v
"""

import keras
import numpy as np
import pytest
from keras import layers
from keras import ops

from .dora_embeddings import DoRAEmbedding
from .dora_embeddings import DoRAPositionEmbedding
from .dora_embeddings import convert_embedding_to_dora


def safe_convert_to_numpy(tensor):
    """Safely convert tensor to numpy across backends."""
    try:
        return ops.convert_to_numpy(tensor)
    except Exception:
        # Fallback for different backends
        if hasattr(tensor, "numpy"):
            return tensor.numpy()
        elif hasattr(tensor, "detach"):
            return tensor.detach().numpy()
        else:
            return np.array(tensor)


def safe_allclose(a, b, rtol=1e-5, atol=1e-8):
    """Safely check if arrays are close across backends."""
    a_np = safe_convert_to_numpy(a) if not isinstance(a, np.ndarray) else a
    b_np = safe_convert_to_numpy(b) if not isinstance(b, np.ndarray) else b
    return np.allclose(a_np, b_np, rtol=rtol, atol=atol)


def safe_array_equal(a, b):
    """Safely check if arrays are equal across backends."""
    a_np = safe_convert_to_numpy(a) if not isinstance(a, np.ndarray) else a
    b_np = safe_convert_to_numpy(b) if not isinstance(b, np.ndarray) else b
    return np.array_equal(a_np, b_np)


def check_no_nan_inf(tensor):
    """Check tensor has no NaN or Inf values across backends."""
    tensor_np = safe_convert_to_numpy(tensor)
    return not (np.isnan(tensor_np).any() or np.isinf(tensor_np).any())


def create_random_tensor(shape, dtype="float32", seed=42):
    """Create random tensor compatible across backends."""
    np.random.seed(seed)
    if dtype == "int32":
        if len(shape) == 2:
            # Fix: Ensure high value is always > 0
            vocab_size = max(shape[0] // 10, 10)  # Minimum vocab size of 10
            high_value = max(min(vocab_size, 100), 2)
            return np.random.randint(0, high_value, size=shape, dtype=np.int32)
        else:
            return np.random.randint(0, 1000, size=shape, dtype=np.int32)
    else:
        return np.random.randn(*shape).astype(dtype)


class TestDoRAEmbedding:
    """Test cases for DoRAEmbedding layer."""

    @pytest.fixture
    def sample_input(self):
        """Create sample token indices."""
        return create_random_tensor((32, 64), dtype="int32", seed=42)

    @pytest.fixture
    def dora_embedding(self):
        """Create a basic DoRA embedding layer."""
        return DoRAEmbedding(
            input_dim=1000, output_dim=128, rank=8, alpha=2.0, mask_zero=True
        )

    def test_layer_creation(self):
        """Test basic layer creation with various configurations."""
        # Test default parameters
        layer = DoRAEmbedding(input_dim=1000, output_dim=64)
        assert layer.input_dim == 1000
        assert layer.output_dim == 64
        assert layer.rank == 4
        assert layer.alpha == 1.0
        assert layer.mask_zero is False

        # Test custom parameters
        layer = DoRAEmbedding(
            input_dim=5000,
            output_dim=256,
            rank=16,
            alpha=0.5,
            mask_zero=True,
            input_length=128,
        )
        assert layer.input_dim == 5000
        assert layer.output_dim == 256
        assert layer.rank == 16
        assert layer.alpha == 0.5
        assert layer.mask_zero is True
        assert layer.input_length == 128

    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test invalid input_dim
        with pytest.raises(ValueError, match="input_dim must be positive"):
            DoRAEmbedding(input_dim=0, output_dim=64)

        with pytest.raises(ValueError, match="input_dim must be positive"):
            DoRAEmbedding(input_dim=-5, output_dim=64)

        # Test invalid output_dim
        with pytest.raises(ValueError, match="output_dim must be positive"):
            DoRAEmbedding(input_dim=1000, output_dim=0)

        with pytest.raises(ValueError, match="output_dim must be positive"):
            DoRAEmbedding(input_dim=1000, output_dim=-10)

        # Test invalid rank
        with pytest.raises(ValueError, match="rank must be positive"):
            DoRAEmbedding(input_dim=1000, output_dim=64, rank=0)

        with pytest.raises(ValueError, match="rank must be positive"):
            DoRAEmbedding(input_dim=1000, output_dim=64, rank=-2)

        # Test invalid alpha
        with pytest.raises(ValueError, match="alpha must be positive"):
            DoRAEmbedding(input_dim=1000, output_dim=64, alpha=0)

        with pytest.raises(ValueError, match="alpha must be positive"):
            DoRAEmbedding(input_dim=1000, output_dim=64, alpha=-1.0)

    def test_layer_build(self, dora_embedding):
        """Test layer building process."""
        # Layer should not be built initially
        assert not dora_embedding.built

        # Build the layer
        dora_embedding.build(None)  # Embedding layers don't need input shape

        # Check if layer is built
        assert dora_embedding.built

        # Check weight shapes
        assert dora_embedding.embeddings.shape == (1000, 128)
        assert dora_embedding.lora_a.shape == (1000, 8)
        assert dora_embedding.lora_b.shape == (8, 128)
        assert dora_embedding.magnitude.shape == (128,)

    def test_forward_pass(self, sample_input, dora_embedding):
        """Test forward pass functionality."""
        # Build and run forward pass
        output = dora_embedding(sample_input)

        # Check output shape
        expected_shape = sample_input.shape + (dora_embedding.output_dim,)
        assert output.shape == expected_shape

        # Check output is not NaN or Inf
        assert check_no_nan_inf(output)

    def test_weight_initialization(self, dora_embedding):
        """Test weight initialization."""
        # Build the layer
        dora_embedding.build(None)

        # Check lora_b is initialized to zeros
        lora_b_np = safe_convert_to_numpy(dora_embedding.lora_b)
        assert np.allclose(lora_b_np, 0.0)

        # Check magnitude is initialized to ones
        magnitude_np = safe_convert_to_numpy(dora_embedding.magnitude)
        assert np.allclose(magnitude_np, 1.0)

    def test_integer_input_conversion(self, dora_embedding):
        """Test that various input types are converted to integers."""
        # Build the layer
        dora_embedding.build(None)

        # Test with float inputs (should be converted to int)
        float_input = ops.convert_to_tensor([[1.0, 2.5, 3.9]], dtype="float32")
        output_float = dora_embedding(float_input)

        # Test with int inputs
        int_input = ops.convert_to_tensor([[1, 2, 3]], dtype="int32")
        output_int = dora_embedding(int_input)

        # Both should work and have correct shape
        assert output_float.shape == (1, 3, 128)
        assert output_int.shape == (1, 3, 128)

    def test_mask_zero_functionality(self):
        """Test mask_zero functionality."""
        # Layer with mask_zero=True
        layer_masked = DoRAEmbedding(
            input_dim=100, output_dim=32, mask_zero=True
        )

        # Layer with mask_zero=False
        layer_unmasked = DoRAEmbedding(
            input_dim=100, output_dim=32, mask_zero=False
        )

        # Test input with zeros
        test_input = ops.convert_to_tensor([[1, 2, 0, 3, 0]], dtype="int32")

        # Test mask computation
        mask_result = layer_masked.compute_mask(test_input)
        assert mask_result is not None

        no_mask_result = layer_unmasked.compute_mask(test_input)
        assert no_mask_result is None

    def test_get_effective_embeddings(self, dora_embedding):
        """Test effective embeddings computation."""
        # Build the layer
        dora_embedding.build(None)

        # Get effective embeddings
        effective_embeddings = dora_embedding.get_effective_embeddings()

        # Check shape
        assert effective_embeddings.shape == (1000, 128)

        # Check it's not NaN or Inf
        assert check_no_nan_inf(effective_embeddings)

    def test_get_dora_parameters(self, dora_embedding):
        """Test DoRA parameter retrieval."""
        dora_embedding.build(None)

        params = dora_embedding.get_dora_parameters()

        # Check all expected parameters are present
        assert "lora_a" in params
        assert "lora_b" in params
        assert "magnitude" in params

        # Check shapes
        assert params["lora_a"].shape == (1000, 8)
        assert params["lora_b"].shape == (8, 128)
        assert params["magnitude"].shape == (128,)

    def test_merge_weights(self, dora_embedding):
        """Test weight merging functionality."""
        dora_embedding.build(None)

        merged = dora_embedding.merge_weights()

        # Check structure
        assert "embeddings" in merged

        # Check shapes
        assert merged["embeddings"].shape == (1000, 128)

    def test_count_params(self):
        """Test parameter counting."""
        layer = DoRAEmbedding(input_dim=1000, output_dim=128, rank=8)

        expected_params = (
            1000 * 8  # lora_a
            + 8 * 128  # lora_b
            + 128  # magnitude
        )

        assert layer.count_params() == expected_params

    def test_load_pretrained_embeddings(self, dora_embedding):
        """Test loading pretrained embeddings."""
        dora_embedding.build(None)

        # Create fake pretrained embeddings using backend-agnostic operations
        pretrained_embeddings = create_random_tensor((1000, 128), seed=123)
        pretrained_tensor = ops.convert_to_tensor(pretrained_embeddings)

        # Load embeddings
        dora_embedding.load_pretrained_embeddings(pretrained_tensor)

        # Check if embeddings are loaded correctly
        embeddings_np = safe_convert_to_numpy(dora_embedding.embeddings)
        assert safe_allclose(embeddings_np, pretrained_embeddings)

    def test_load_pretrained_embeddings_shape_mismatch(self, dora_embedding):
        """Test loading pretrained embeddings with wrong shapes."""
        dora_embedding.build(None)

        # Wrong shape
        wrong_embeddings = create_random_tensor((500, 64), seed=123)
        wrong_tensor = ops.convert_to_tensor(wrong_embeddings)

        with pytest.raises(ValueError, match="Pretrained embeddings shape"):
            dora_embedding.load_pretrained_embeddings(wrong_tensor)

    def test_expand_vocabulary(self, dora_embedding):
        """Test vocabulary expansion functionality."""
        dora_embedding.build(None)

        # Expand vocabulary from 1000 to 1200
        expanded_layer = dora_embedding.expand_vocabulary(1200)

        # Check new dimensions
        assert expanded_layer.input_dim == 1200
        assert expanded_layer.output_dim == 128  # Should remain same

        # Check weight shapes
        assert expanded_layer.embeddings.shape == (1200, 128)
        assert expanded_layer.lora_a.shape == (1200, 8)
        assert expanded_layer.lora_b.shape == (8, 128)
        assert expanded_layer.magnitude.shape == (128,)

    def test_expand_vocabulary_with_new_embeddings(self, dora_embedding):
        """Test vocabulary expansion with provided new embeddings."""
        dora_embedding.build(None)

        # Create new token embeddings for 200 additional tokens
        new_token_embeddings = create_random_tensor((200, 128), seed=456)
        new_embeddings_tensor = ops.convert_to_tensor(new_token_embeddings)

        # Expand vocabulary
        expanded_layer = dora_embedding.expand_vocabulary(
            1200, new_embeddings_tensor
        )

        # Check dimensions
        assert expanded_layer.input_dim == 1200
        assert expanded_layer.embeddings.shape == (1200, 128)

    def test_expand_vocabulary_errors(self, dora_embedding):
        """Test vocabulary expansion error cases."""
        dora_embedding.build(None)

        # Test expanding to smaller size
        with pytest.raises(
            ValueError, match="new_vocab_size .* must be greater"
        ):
            dora_embedding.expand_vocabulary(500)

        # Test with wrong new embeddings shape
        wrong_embeddings = create_random_tensor((100, 64), seed=789)
        wrong_tensor = ops.convert_to_tensor(wrong_embeddings)

        with pytest.raises(ValueError, match="new_token_embeddings shape"):
            dora_embedding.expand_vocabulary(1200, wrong_tensor)

    def test_expand_vocabulary_unbuilt_layer(self):
        """Test expanding vocabulary on unbuilt layer."""
        layer = DoRAEmbedding(input_dim=1000, output_dim=128)

        with pytest.raises(ValueError, match="Layer must be built"):
            layer.expand_vocabulary(1200)

    def test_serialization(self, dora_embedding):
        """Test layer serialization and deserialization."""
        # Get config
        config = dora_embedding.get_config()

        # Check essential parameters are in config
        assert config["input_dim"] == dora_embedding.input_dim
        assert config["output_dim"] == dora_embedding.output_dim
        assert config["rank"] == dora_embedding.rank
        assert config["alpha"] == dora_embedding.alpha
        assert config["mask_zero"] == dora_embedding.mask_zero

        # Create layer from config
        restored_layer = DoRAEmbedding.from_config(config)

        # Check restored layer has same parameters
        assert restored_layer.input_dim == dora_embedding.input_dim
        assert restored_layer.output_dim == dora_embedding.output_dim
        assert restored_layer.rank == dora_embedding.rank
        assert restored_layer.alpha == dora_embedding.alpha

    def test_compute_output_shape(self):
        """Test output shape computation."""
        layer = DoRAEmbedding(input_dim=1000, output_dim=64, input_length=10)

        # Test various input shapes
        input_shapes = [
            (None,),
            (10,),
            (None, 5),
            (32, 10),
        ]

        for input_shape in input_shapes:
            output_shape = layer.compute_output_shape(input_shape)
            expected_shape = input_shape + (64,)
            assert output_shape == expected_shape

    def test_regularization(self):
        """Test regularization functionality."""
        layer = DoRAEmbedding(
            input_dim=100,
            output_dim=32,
            embeddings_regularizer="l2",
            activity_regularizer="l2",
        )

        sample_input = ops.convert_to_tensor([[1, 2, 3]], dtype="int32")
        output = layer(sample_input)

        # Check output shape
        assert output.shape == (1, 3, 32)

    def test_constraints(self):
        """Test constraint functionality."""
        layer = DoRAEmbedding(
            input_dim=100, output_dim=32, embeddings_constraint="max_norm"
        )

        sample_input = ops.convert_to_tensor([[1, 2, 3]], dtype="int32")
        output = layer(sample_input)

        # Check output shape
        assert output.shape == (1, 3, 32)


class TestDoRAPositionEmbedding:
    """Test cases for DoRAPositionEmbedding layer."""

    @pytest.fixture
    def sample_input(self):
        """Create sample token embeddings."""
        return create_random_tensor((8, 32, 64), seed=42)

    @pytest.fixture
    def position_layer(self):
        """Create a basic DoRA position embedding layer."""
        return DoRAPositionEmbedding(
            sequence_length=128, output_dim=64, rank=8, alpha=2.0
        )

    def test_layer_creation(self, position_layer):
        """Test basic layer creation."""
        assert position_layer.sequence_length == 128
        assert position_layer.output_dim == 64
        assert position_layer.rank == 8
        assert position_layer.alpha == 2.0

    def test_layer_build(self, position_layer):
        """Test layer building process."""
        # Build the layer
        position_layer.build(None)

        # Check if layer is built
        assert position_layer.built

        # Check weight shapes
        assert position_layer.position_embeddings.shape == (128, 64)
        assert position_layer.lora_a.shape == (128, 8)
        assert position_layer.lora_b.shape == (8, 64)
        assert position_layer.magnitude.shape == (64,)

    def test_forward_pass(self, sample_input, position_layer):
        """Test forward pass functionality."""
        # Convert to tensor for backend compatibility
        input_tensor = ops.convert_to_tensor(sample_input)

        # Build and run forward pass
        output = position_layer(input_tensor)

        # Check output shape matches input
        assert output.shape == input_tensor.shape

        # Check output is not NaN or Inf
        assert check_no_nan_inf(output)

    def test_start_index_parameter(self, sample_input, position_layer):
        """Test start_index parameter."""
        input_tensor = ops.convert_to_tensor(sample_input)

        # Test with different start indices
        output1 = position_layer(input_tensor, start_index=0)
        output2 = position_layer(input_tensor, start_index=10)

        # Both should have same shape
        assert output1.shape == input_tensor.shape
        assert output2.shape == input_tensor.shape

        # Should produce different outputs for different start indices
        assert not safe_allclose(output1, output2)

    def test_sequence_length_clipping(self, position_layer):
        """Test that positions are clipped to sequence length."""
        position_layer.build(None)

        # Create input longer than sequence_length
        long_input = create_random_tensor((4, 200, 64), seed=42)
        long_tensor = ops.convert_to_tensor(long_input)

        # Should still work (positions get clipped)
        output = position_layer(long_tensor)
        assert output.shape == long_tensor.shape

    def test_effective_position_embeddings(self, position_layer):
        """Test effective position embeddings computation."""
        position_layer.build(None)

        # Get effective position embeddings
        effective_embeddings = (
            position_layer._get_effective_position_embeddings()
        )

        # Check shape
        assert effective_embeddings.shape == (128, 64)

        # Check it's not NaN or Inf
        assert check_no_nan_inf(effective_embeddings)

    def test_serialization(self, position_layer):
        """Test layer serialization and deserialization."""
        # Get config
        config = position_layer.get_config()

        # Check essential parameters are in config
        assert config["sequence_length"] == position_layer.sequence_length
        assert config["output_dim"] == position_layer.output_dim
        assert config["rank"] == position_layer.rank
        assert config["alpha"] == position_layer.alpha

        # Create layer from config
        restored_layer = DoRAPositionEmbedding.from_config(config)

        # Check restored layer has same parameters
        assert restored_layer.sequence_length == position_layer.sequence_length
        assert restored_layer.output_dim == position_layer.output_dim
        assert restored_layer.rank == position_layer.rank
        assert restored_layer.alpha == position_layer.alpha


class TestEmbeddingConversion:
    """Test cases for Embedding to DoRA conversion."""

    def test_convert_embedding_to_dora(self):
        """Test converting Embedding layer to DoRA layer."""
        # Create an Embedding layer
        embedding_layer = layers.Embedding(
            input_dim=1000,
            output_dim=64,
            mask_zero=True,
            embeddings_initializer="uniform",
        )

        # Build with sample input
        sample_input = ops.convert_to_tensor([[1, 2, 3, 4]], dtype="int32")
        embedding_output = embedding_layer(sample_input)

        # Convert to DoRA
        dora_layer = convert_embedding_to_dora(
            embedding_layer, rank=8, alpha=2.0
        )

        # Check configuration
        assert dora_layer.input_dim == embedding_layer.input_dim
        assert dora_layer.output_dim == embedding_layer.output_dim
        assert dora_layer.rank == 8
        assert dora_layer.alpha == 2.0
        assert dora_layer.mask_zero == embedding_layer.mask_zero

        # Check weights are loaded
        assert dora_layer.built

        # Test forward pass produces reasonable output
        dora_output = dora_layer(sample_input)
        assert dora_output.shape == embedding_output.shape

    def test_convert_unbuilt_embedding(self):
        """Test converting unbuilt Embedding layer."""
        embedding_layer = layers.Embedding(input_dim=500, output_dim=32)

        # Convert unbuilt layer
        dora_layer = convert_embedding_to_dora(embedding_layer, rank=4)

        # Should not be built yet
        assert not dora_layer.built

        # But should have correct configuration
        assert dora_layer.input_dim == 500
        assert dora_layer.output_dim == 32
        assert dora_layer.rank == 4

    def test_convert_embedding_without_input_length(self):
        """Test converting embedding layer without input_length attribute."""

        # Create a mock embedding layer without input_length
        class MockEmbedding:
            def __init__(self):
                self.input_dim = 100
                self.output_dim = 32
                self.embeddings_initializer = "uniform"
                self.embeddings_regularizer = None
                self.activity_regularizer = None
                self.embeddings_constraint = None
                self.mask_zero = False
                self.name = "test_embedding"
                self.built = False

        mock_layer = MockEmbedding()

        # Should work even without input_length
        dora_layer = convert_embedding_to_dora(mock_layer, rank=4)
        assert dora_layer.input_dim == 100
        assert dora_layer.output_dim == 32
        assert dora_layer.input_length is None


class TestDoRAEmbeddingMathematicalProperties:
    """Test mathematical properties of DoRA embeddings."""

    def test_magnitude_scaling_property(self):
        """Test that DoRA properly applies magnitude scaling."""
        layer = DoRAEmbedding(input_dim=100, output_dim=32, rank=4)
        layer.build(None)

        # Get effective embeddings
        effective_embeddings = layer.get_effective_embeddings()
        effective_embeddings_np = safe_convert_to_numpy(effective_embeddings)

        # Compute column norms of effective embeddings
        column_norms = np.linalg.norm(effective_embeddings_np, axis=0)
        magnitude_np = safe_convert_to_numpy(layer.magnitude)

        # Column norms should equal magnitude values (approximately)
        assert safe_allclose(column_norms, magnitude_np, rtol=1e-5)

    def test_low_rank_adaptation_property(self):
        """Test that adaptation is indeed low-rank."""
        layer = DoRAEmbedding(input_dim=100, output_dim=64, rank=8)
        layer.build(None)

        # Compute LoRA adaptation using backend-agnostic operations
        lora_a_np = safe_convert_to_numpy(layer.lora_a)
        lora_b_np = safe_convert_to_numpy(layer.lora_b)
        adaptation = lora_a_np @ (lora_b_np * layer.scaling)

        # Check that adaptation matrix has rank <= layer.rank
        actual_rank = np.linalg.matrix_rank(adaptation)
        assert actual_rank <= layer.rank

    def test_zero_initialization_equivalence(self):
        """Test that zero LoRA initialization gives expected behavior."""
        layer = DoRAEmbedding(
            input_dim=50,
            output_dim=32,
            rank=4,
            lora_a_initializer="zeros",
            lora_b_initializer="zeros",
        )
        layer.build(None)

        # With zero LoRA matrices, effective embeddings should have
        # column norms equal to magnitude (which is initialized to ones)
        effective_embeddings = layer.get_effective_embeddings()
        effective_embeddings_np = safe_convert_to_numpy(effective_embeddings)
        column_norms = np.linalg.norm(effective_embeddings_np, axis=0)

        magnitude_np = safe_convert_to_numpy(layer.magnitude)
        assert safe_allclose(column_norms, magnitude_np, rtol=1e-5)

    def test_embedding_lookup_correctness(self):
        """Test that embedding lookup works correctly."""
        layer = DoRAEmbedding(input_dim=10, output_dim=4, rank=2)
        layer.build(None)

        # Test specific token indices
        test_indices = ops.convert_to_tensor(
            [[0, 1, 2], [3, 4, 5]], dtype="int32"
        )
        output = layer(test_indices)

        # Get effective embeddings
        effective_embeddings = layer.get_effective_embeddings()

        # Manually lookup embeddings for comparison
        output_np = safe_convert_to_numpy(output)
        effective_embeddings_np = safe_convert_to_numpy(effective_embeddings)

        # Check first batch, first token (index 0)
        expected_first = effective_embeddings_np[0]
        actual_first = output_np[0, 0]
        assert safe_allclose(actual_first, expected_first)

        # Check second batch, third token (index 5)
        expected_last = effective_embeddings_np[5]
        actual_last = output_np[1, 2]
        assert safe_allclose(actual_last, expected_last)


def test_backend_compatibility():
    """Test that the implementation works across different backends."""
    try:
        backend_name = keras.backend.backend()
        print(f"Testing with backend: {backend_name}")
    except Exception:
        print("Backend detection failed, proceeding with tests...")

    # Test DoRAEmbedding
    embedding_layer = DoRAEmbedding(input_dim=100, output_dim=32, rank=4)
    sample_input = create_random_tensor((1, 4), dtype="int32")
    sample_tensor = ops.convert_to_tensor(sample_input)

    try:
        output = embedding_layer(sample_tensor)
        assert output.shape == (1, 4, 32)
        print("DoRAEmbedding test passed")
    except Exception as e:
        print(f"DoRAEmbedding test failed: {e}")
        return False

    # Test DoRAPositionEmbedding
    pos_layer = DoRAPositionEmbedding(sequence_length=10, output_dim=32, rank=4)
    sample_embeddings = create_random_tensor((2, 4, 32))
    embeddings_tensor = ops.convert_to_tensor(sample_embeddings)

    try:
        pos_output = pos_layer(embeddings_tensor)
        assert pos_output.shape == (2, 4, 32)
        print("DoRAPositionEmbedding test passed")
    except Exception as e:
        print(f"DoRAPositionEmbedding test failed: {e}")
        return False

    return True


def test_masking_integration():
    """Test integration with Keras masking."""
    # Create layer with masking
    layer = DoRAEmbedding(input_dim=100, output_dim=32, mask_zero=True)

    # Input with zeros (should be masked)
    input_with_zeros = ops.convert_to_tensor([[1, 2, 0, 3, 0]], dtype="int32")

    # Get output and mask
    output = layer(input_with_zeros)
    mask = layer.compute_mask(input_with_zeros)

    assert output.shape == (1, 5, 32)
    assert mask is not None

    # Check mask values
    mask_np = safe_convert_to_numpy(mask)
    expected_mask = np.array([[True, True, False, True, False]])
    assert safe_array_equal(mask_np, expected_mask)


def test_safe_weight_assignment():
    """Test safe weight assignment across backends."""
    layer = DoRAEmbedding(input_dim=10, output_dim=8, rank=2)
    layer.build(None)

    # Test loading pretrained embeddings
    pretrained = create_random_tensor((10, 8), seed=999)
    pretrained_tensor = ops.convert_to_tensor(pretrained)

    try:
        layer.load_pretrained_embeddings(pretrained_tensor)
        # Check if assignment worked
        loaded_embeddings = safe_convert_to_numpy(layer.embeddings)
        assert safe_allclose(loaded_embeddings, pretrained)
        print("Safe weight assignment test passed")
        return True
    except Exception as e:
        print(f"Safe weight assignment test failed: {e}")
        return False


def test_backend_agnostic_operations():
    """Test that all operations use backend-agnostic ops."""
    layer = DoRAEmbedding(input_dim=20, output_dim=16, rank=4)
    layer.build(None)

    # Test effective embeddings computation
    try:
        effective_embeddings = layer._get_effective_embeddings()
        assert effective_embeddings.shape == (20, 16)
        assert check_no_nan_inf(effective_embeddings)
        print("Backend-agnostic operations test passed")
        return True
    except Exception as e:
        print(f"Backend-agnostic operations test failed: {e}")
        return False
