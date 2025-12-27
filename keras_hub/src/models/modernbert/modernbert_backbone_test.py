import pytest
from keras import ops
from keras_hub.src.models.modernbert.modernbert_backbone import ModernBertBackbone
from keras_hub.src.tests.test_case import TestCase

class ModernBertBackboneTest(TestCase):
    """
    Tests for ModernBERT backbone model.
    """

    def setUp(self):
        self.init_kwargs = {
            "vocabulary_size": 10,
            "num_layers": 4, # Use at least 3 layers to test alternating attention
            "num_heads": 2,
            "hidden_dim": 8,
            "intermediate_dim": 16,
            "dropout": 0.0,
            "local_attention_window": 2,
            "global_attn_every_n_layers": 3,
        }
        self.input_data = {
            "token_ids": ops.cast(ops.ones((2, 5)), dtype="int32"),
            "padding_mask": ops.cast(ops.ones((2, 5)), dtype="int32"),
        }

    def test_backbone_basics(self):
        """
        Verify the backbone runs and produces correct output shapes.
        """
        self.run_backbone_test(
            cls=ModernBertBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 8),
        )

    @pytest.mark.large
    def test_saved_model(self):
        """
        Verify the model can be saved and loaded accurately.
        """
        self.run_model_saving_test(
            cls=ModernBertBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )

    def test_alternating_attention_config(self):
        """
        Verify that layers are correctly assigned local vs global attention.
        """
        model = ModernBertBackbone(**self.init_kwargs)
        # In ModernBERT, layers are 1-indexed for the 'every n' logic usually.
        # Layer 3 (index 2) should be Global if global_attn_every_n_layers=3.
        # We check the 'is_global' attribute we implemented in the EncoderLayer.
        layers = [l for l in model.layers if "modern_bert_encoder_layer" in l.name]
        
        # Test routing logic: Layer index 0, 1 -> Local; Index 2 -> Global
        self.assertFalse(layers[0].is_global)
        self.assertFalse(layers[1].is_global)
        self.assertTrue(layers[2].is_global)

    def test_variable_sequence_length(self):
        """
        Ensure the backbone handles different sequence lengths via RoPE.
        """
        model = ModernBertBackbone(**self.init_kwargs)
        # Test with a longer length to ensure RoPE indices are generated dynamically
        longer_input = {
            "token_ids": ops.cast(ops.ones((1, 12)), dtype="int32"),
            "padding_mask": ops.cast(ops.ones((1, 12)), dtype="int32"),
        }
        output = model(longer_input)
        self.assertEqual(ops.shape(output), (1, 12, 8))

    def test_mixed_precision(self):
        """
        Verify the backbone works correctly with mixed precision policies.
        """
        self.run_backbone_test(
            cls=ModernBertBackbone,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output_shape=(2, 5, 8),
            run_mixed_precision_check=True,
        )