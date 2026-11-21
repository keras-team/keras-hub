import keras
import numpy as np
from keras import ops

from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_hub.src.models.modern_bert.modern_bert_layers import (
    ModernBertAttention,
)
from keras_hub.src.models.modern_bert.modern_bert_layers import (
    ModernBertEncoderLayer,
)
from keras_hub.src.models.modern_bert.modern_bert_layers import ModernBertMLP
from keras_hub.src.tests.test_case import TestCase


class ModernBertLayersTest(TestCase):
    """Tests for ModernBERT specific layers."""

    def setUp(self):
        self.compute_dtype = keras.config.dtype_policy().compute_dtype

    def _rotary_embedding(self):
        return RotaryEmbedding(max_wavelength=10000)

    def test_attention_layer(self):
        self.run_layer_test(
            cls=ModernBertAttention,
            init_kwargs={
                "hidden_dim": 16,
                "num_heads": 2,
                "rotary_embedding": self._rotary_embedding(),
                "local_attention_window": 128,
            },
            input_data=ops.ones(
                (2, 4, 16),
                dtype=self.compute_dtype,
            ),
            expected_output_shape=(2, 4, 16),
            expected_num_trainable_weights=2,
        )

    def test_attention_requires_divisible_hidden_dim(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            ModernBertAttention(hidden_dim=15, num_heads=2)

    def test_mlp_layer(self):
        self.run_layer_test(
            cls=ModernBertMLP,
            init_kwargs={
                "hidden_dim": 16,
                "intermediate_dim": 32,
            },
            input_data=ops.ones(
                (2, 4, 16),
                dtype=self.compute_dtype,
            ),
            expected_output_shape=(2, 4, 16),
            expected_num_trainable_weights=3,
        )

    def test_encoder_layer(self):
        self.run_layer_test(
            cls=ModernBertEncoderLayer,
            init_kwargs={
                "hidden_dim": 16,
                "intermediate_dim": 32,
                "num_heads": 2,
                "layer_idx": 1,
                "rotary_embedding": self._rotary_embedding(),
                "local_attention_window": 128,
            },
            input_data=ops.ones(
                (2, 4, 16),
                dtype=self.compute_dtype,
            ),
            expected_output_shape=(2, 4, 16),
            expected_num_trainable_weights=7,
        )

    def test_encoder_layer_zero_has_identity_attention_norm(self):
        """ModernBERT omits the attention norm on layer 0."""
        layer_zero = ModernBertEncoderLayer(
            hidden_dim=16,
            intermediate_dim=32,
            num_heads=2,
            layer_idx=0,
        )
        layer_one = ModernBertEncoderLayer(
            hidden_dim=16,
            intermediate_dim=32,
            num_heads=2,
            layer_idx=1,
        )

        self.assertIsInstance(layer_zero.attn_norm, keras.layers.Identity)
        self.assertIsInstance(
            layer_one.attn_norm, keras.layers.LayerNormalization
        )

    def test_attention_with_padding_mask(self):
        attention = ModernBertAttention(
            hidden_dim=16,
            num_heads=2,
            local_attention_window=128,
            rotary_embedding=self._rotary_embedding(),
        )

        x = ops.ones(
            (1, 4, 16),
            dtype=self.compute_dtype,
        )

        padding_mask = ops.convert_to_tensor(
            [[1, 1, 0, 0]],
            dtype="int32",
        )

        output = attention(
            x,
            padding_mask=padding_mask,
        )

        output_np = ops.convert_to_numpy(output)

        self.assertFalse(np.any(np.isnan(output_np)))

    def test_sliding_window_mask(self):
        attention = ModernBertAttention(
            hidden_dim=8,
            num_heads=2,
            local_attention_window=2,
            rotary_embedding=self._rotary_embedding(),
        )

        mask = attention._get_sliding_window_mask(
            seq_len=4,
            dtype="float32",
        )

        expected = [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
        ]

        self.assertAllClose(mask, expected)
