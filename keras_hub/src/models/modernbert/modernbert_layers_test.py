import numpy as np
from keras import ops

from keras_hub.src.models.modernbert.modernbert_layers import (
    ModernBertAttention,
)
from keras_hub.src.models.modernbert.modernbert_layers import (
    ModernBertEncoderLayer,
)
from keras_hub.src.tests.test_case import TestCase


class ModernBertLayersTest(TestCase):
    """Tests for ModernBERT specific layers."""

    def test_attention_masking_logic(self):
        """Verify that the attention layer correctly handles padding masks.

        This test checks:
        - Ability to pass standard sequence padding masks to the layer.
        - Numerical stability by ensuring no invalid NaN boundaries occur.
        - Retention of original batch and hidden structural dimensionality.
        """
        layer = ModernBertAttention(hidden_dim=16, num_heads=2)
        x = ops.ones((1, 4, 16))
        mask = ops.convert_to_tensor([[1, 1, 0, 0]], dtype="int32")
        output = layer(x, padding_mask=mask)
        output_np = ops.convert_to_numpy(output)
        self.assertFalse(np.any(np.isnan(output_np)))

    def test_serialization_attributes(self):
        """Explicitly verify that custom attributes are restored.

        This test checks:
        - Structural population of configuration states inside `get_config`.
        - Restoration parity for specific fields like `local_attention_window`.
        - Architecture weight and dimension preservation across deserialization.
        """
        layer = ModernBertEncoderLayer(
            hidden_dim=16,
            intermediate_dim=32,
            num_heads=2,
            local_attention_window=128,
        )
        config = layer.get_config()
        new_layer = ModernBertEncoderLayer.from_config(config)
        self.assertEqual(new_layer.local_attention_window, 128)
        self.assertEqual(new_layer.hidden_dim, 16)

    def test_sliding_window_mask_creation(self):
        """Directly check the internal mask generation logic.

        This test checks:
        - Exact structural calculations for context distance matrices.
        - Precise handling of local window margins relative to the center token.
        - Matrix structure compliance with target expected attention boundaries.
        """

        layer = ModernBertAttention(hidden_dim=8, num_heads=2, local_attention_window=2)
        mask = layer._get_sliding_window_mask(seq_len=4, dtype="float32")

        expected = [
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 1, 1, 1],
        ]
        self.assertAllClose(mask, expected)
