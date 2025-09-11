import pytest
from keras import mixed_precision
from keras import ops

from keras_hub.src.models.voxtral.voxtral_backbone import VoxTralBackbone
from keras_hub.src.tests.test_case import TestCase


class VoxTralBackboneTest(TestCase):
    """Unit tests for VoxTralBackbone."""

    def setUp(self):
        """Initialize default backbone arguments and input data."""
        self.init_kwargs = {
            "num_layers": 2,
            "num_heads": 2,
            "hidden_dim": 16,
            "intermediate_dim": 32,
            "adapter_downsample": 2,
            "dropout": 0.0,
            "max_chunk_seconds": 1,
            "sr": 16000,
            "hop_length": 160,
            "dtype": "float32",
        }
        # Dummy input: shape (batch, time, features)
        self.input_data = ops.ones((1, 2542, 128), dtype="float32")

    def test_backbone_basics(self):
        """Test forward pass and output shape with float32."""
        mixed_precision.set_global_policy("float32")
        model = VoxTralBackbone(**self.init_kwargs)
        output = model(self.input_data)
        assert tuple(output.shape) == (1, 650, 16)
        assert output.dtype.name == "float32"

    @pytest.mark.large
    def test_saved_model(self):
        """Test saving and loading the model."""
        self.run_model_saving_test(
            cls=VoxTralBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )
