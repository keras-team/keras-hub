import numpy as np
from keras import ops

from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_hub.src.models.modernbert.modernbert_layers import (
    ModernBertAttention,
)
from keras_hub.src.models.modernbert.modernbert_layers import (
    ModernBertEncoderLayer,
)
from keras_hub.src.models.modernbert.modernbert_layers import ModernBertMLP
from keras_hub.src.tests.test_case import TestCase


class ModernBertLayersTest(TestCase):
    """Tests for ModernBERT specific layers."""

    def test_layer_behaviors(self):
        rotary_emb = RotaryEmbedding(max_wavelength=10000)

        self.run_layer_test(
            cls=ModernBertAttention,
            init_kwargs={
                "hidden_dim": 16,
                "num_heads": 2,
                "rotary_embedding": rotary_emb,
                "local_attention_window": 128,
            },
            input_data=ops.ones((2, 4, 16)),
            expected_output_shape=(2, 4, 16),
            expected_num_trainable_weights=2,
        )

        self.run_layer_test(
            cls=ModernBertMLP,
            init_kwargs={
                "hidden_dim": 16,
                "intermediate_dim": 32,
            },
            input_data=ops.ones((2, 4, 16)),
            expected_output_shape=(2, 4, 16),
            expected_num_trainable_weights=3,
        )

        self.run_layer_test(
            cls=ModernBertEncoderLayer,
            init_kwargs={
                "hidden_dim": 16,
                "intermediate_dim": 32,
                "num_heads": 2,
                "layer_idx": 1,
                "rotary_embedding": rotary_emb,
                "local_attention_window": 128,
            },
            input_data=ops.ones((2, 4, 16)),
            expected_output_shape=(2, 4, 16),
            expected_num_trainable_weights=7,
        )

    def test_attention_masking_logic(self):
        rotary_emb = RotaryEmbedding(max_wavelength=10000)

        layer = ModernBertAttention(
            hidden_dim=16,
            num_heads=2,
            local_attention_window=128,
            rotary_embedding=rotary_emb,
        )
        x = ops.ones((1, 4, 16))
        mask = ops.convert_to_tensor([[1, 1, 0, 0]], dtype="int32")
        output = layer(x, padding_mask=mask)
        output_np = ops.convert_to_numpy(output)
        self.assertFalse(np.any(np.isnan(output_np)))

    def test_sliding_window_mask_creation(self):
        rotary_emb = RotaryEmbedding(max_wavelength=10000)
        layer = ModernBertAttention(
            hidden_dim=8,
            num_heads=2,
            local_attention_window=2,
            rotary_embedding=rotary_emb,
        )
        mask = layer._get_sliding_window_mask(seq_len=4, dtype="float32")

        expected = [
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 1, 1, 1],
        ]
        self.assertAllClose(mask, expected)
