import keras
import numpy as np
from keras import ops

from keras_hub.src.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_hub.src.models.modernbert.modern_bert_layers import (
    ModernBertAttention,
)
from keras_hub.src.models.modernbert.modern_bert_layers import (
    ModernBertEncoderLayer,
)
from keras_hub.src.models.modernbert.modern_bert_layers import ModernBertMLP
from keras_hub.src.tests.test_case import TestCase


class ModernBertLayersTest(TestCase):
    """Tests for ModernBERT specific layers."""

    def test_layer_behaviors(self):
        # Keep this test in float32 because run_layer_test() checks
        # output dtype against layer.dtype.
        previous_policy = keras.config.dtype_policy()
        keras.config.set_dtype_policy("float32")
        try:
            compute_dtype = keras.config.dtype_policy().compute_dtype

            # ModernBertAttention
            self.run_layer_test(
                cls=ModernBertAttention,
                init_kwargs={
                    "hidden_dim": 16,
                    "num_heads": 2,
                    "rotary_embedding": RotaryEmbedding(
                        max_wavelength=10000,
                        dtype=keras.config.dtype_policy(),
                    ),
                    "local_attention_window": 128,
                },
                input_data=ops.ones(
                    (2, 4, 16),
                    dtype=compute_dtype,
                ),
                expected_output_shape=(2, 4, 16),
                expected_num_trainable_weights=2,
            )

            # ModernBertMLP
            self.run_layer_test(
                cls=ModernBertMLP,
                init_kwargs={
                    "hidden_dim": 16,
                    "intermediate_dim": 32,
                },
                input_data=ops.ones(
                    (2, 4, 16),
                    dtype=compute_dtype,
                ),
                expected_output_shape=(2, 4, 16),
                expected_num_trainable_weights=3,
            )

            # ModernBertEncoderLayer
            self.run_layer_test(
                cls=ModernBertEncoderLayer,
                init_kwargs={
                    "hidden_dim": 16,
                    "intermediate_dim": 32,
                    "num_heads": 2,
                    "layer_idx": 1,
                    "rotary_embedding": RotaryEmbedding(
                        max_wavelength=10000,
                        dtype=keras.config.dtype_policy(),
                    ),
                    "local_attention_window": 128,
                },
                input_data=ops.ones(
                    (2, 4, 16),
                    dtype=compute_dtype,
                ),
                expected_output_shape=(2, 4, 16),
                expected_num_trainable_weights=7,
            )

            # Attention masking logic
            attention = ModernBertAttention(
                hidden_dim=16,
                num_heads=2,
                local_attention_window=128,
                rotary_embedding=RotaryEmbedding(
                    max_wavelength=10000,
                    dtype=keras.config.dtype_policy(),
                ),
            )

            x = ops.ones(
                (1, 4, 16),
                dtype=compute_dtype,
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

            # Sliding window mask creation
            attention = ModernBertAttention(
                hidden_dim=8,
                num_heads=2,
                local_attention_window=2,
                rotary_embedding=RotaryEmbedding(
                    max_wavelength=10000,
                    dtype=keras.config.dtype_policy(),
                ),
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
        finally:
            keras.config.set_dtype_policy(previous_policy)
