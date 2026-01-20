import numpy as np
import os
import pytest
import keras
from keras import ops
from keras_hub.src.tests.test_case import TestCase

from keras_hub.src.models.modernbert.modernbert_layers import (
     ModernBertAttention, ModernBertEncoderLayer,
)
from keras_hub.src.models.modernbert.modernbert_masked_lm import (
    ModernBertMaskedLM,
)

class ModernBertLayersTest(TestCase):
    def test_attention_masking_logic(self):
        """Verify that the attention layer correctly handles padding masks."""
        layer = ModernBertAttention(hidden_dim=16, num_heads=2)
        x = ops.ones((1, 4, 16))
        mask = ops.convert_to_tensor([[1, 1, 0, 0]], dtype="int32")
        output = layer(x, padding_mask=mask)
        
        self.assertFalse(np.any(np.isnan(output)))
        self.assertEqual(output.shape, (1, 4, 16))

    def test_serialization_attributes(self):
        """Explicitly verify that custom attributes are restored."""
        layer = ModernBertEncoderLayer(
            hidden_dim=16, 
            intermediate_dim=32, 
            num_heads=2, 
            local_attention_window=64
        )
        config = layer.get_config()
        new_layer = ModernBertEncoderLayer.from_config(config)
        self.assertEqual(new_layer.local_attention_window, 64)
        self.assertEqual(new_layer.hidden_dim, 16)

    def test_sliding_window_mask_creation(self):
        """Directly check the internal mask generation logic."""
        layer = ModernBertAttention(
            hidden_dim=8, 
            num_heads=2, 
            local_attention_window=2
        )
        mask = layer._get_sliding_window_mask(seq_len=4, dtype="float32")
        expected = [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
        ]
        self.assertAllClose(mask, expected)

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=ModernBertMaskedLM,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )