import keras
import pytest
from keras import ops

from keras_hub.src.models.modernbert.modernbert_backbone import (
    ModernBertBackbone
)
from keras_hub.src.tests.test_case import TestCase

class ModernBertBackboneTest(TestCase):
    """Tests for ModernBERT backbone."""
    
    def setUp(self):
        """Set up a small configuration for testing."""
        self.init_kwargs = {
            "vocabulary_size": 10,
            "hidden_dim": 8,
            "intermediate_dim": 64,
            "num_layers": 2,
            "num_heads": 4,
            "local_attention_window": 128,
            "global_attn_every_n_layers": 2,
            "dropout": 0.0,
        }
        self.input_data = {
            "token_ids": ops.ones((2, 5), dtype="int32"),
            "padding_mask": ops.ones((2, 5), dtype="int32"),
        }

    def test_backbone_basics(self):
        """Test backbone forward pass and standard KerasHub lifecycle.

        This validates:
        1. Forward pass with the given input data.
        2. Config serialization (get_config/from_config).
        3. Model saving and loading via the `.keras` format.
        4. Backend-agnostic execution.
        """
        self.run_backbone_test(
            cls=ModernBertBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 8),
        )

    @pytest.mark.large
    def test_saved_model(self):
        """Test that the model can be saved and loaded in the .keras format."""
        self.run_model_saving_test(
            cls=ModernBertBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )