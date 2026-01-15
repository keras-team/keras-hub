import keras
import pytest
from keras import ops

from keras_hub.src.models.modernbert.modernbert_backbone import ModernBertBackbone
from keras_hub.src.tests.test_case import TestCase

class ModernBertBackboneTest(TestCase):
    """Tests for ModernBERT backbone."""

    def setUp(self):
        """Set up a small configuration for testing."""
        self.init_kwargs = {
            "vocabulary_size": 10,
            "hidden_dim": 8,
            "intermediate_dim": 32,
            "num_layers": 2,
            "num_heads": 4,
            "local_attention_window": 32,
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

    def test_alternating_attention_logic(self):
        """Verify the interleaving of local and global attention layers.

        ModernBERT should alternate between local attention (sliding window)
        and global attention based on the `global_attn_every_n_layers` 
        parameter.
        """
        conf = {**self.init_kwargs, "global_attn_every_n_layers": 2}
        model = ModernBertBackbone(**conf)

        # Layer 1 (index 0): (0 + 1) % 2 != 0 -> Local
        # Layer 2 (index 1): (1 + 1) % 2 == 0 -> Global
        layer0 = model.get_layer("transformer_layer_0")
        layer1 = model.get_layer("transformer_layer_1")

        self.assertEqual(layer0.local_attention_window, 32)
        self.assertIsNone(layer1.local_attention_window)

    def test_variable_sequence_length(self):
        """Ensure the backbone handles dynamic sequence lengths."""
        model = ModernBertBackbone(**self.init_kwargs)
        short_input = {
            "token_ids": ops.ones((1, 3), dtype="int32"),
            "padding_mask": ops.ones((1, 3), dtype="int32"),
        }
        output = model(short_input)
        self.assertEqual(output.shape, (1, 3, 8))

    @pytest.mark.large
    def test_mixed_precision(self):
        """Verify the backbone supports mixed precision training.

        This ensures that the model correctly downcasts to the compute dtype
        specified by the policy (e.g., float16) and handles numerical 
        stability without crashing.
        """
        self.run_precision_test(
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