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
        # Manual test as it is a composite layer
        encoder = ModernBertEncoderLayer(
            hidden_dim=16,
            intermediate_dim=32,
            num_heads=2,
            layer_idx=1,
            rotary_embedding=RotaryEmbedding(
                max_wavelength=10000,
                dtype=keras.config.dtype_policy(),
            ),
            local_attention_window=128,
        )

        inputs = ops.ones(
            (2, 4, 16),
            dtype=compute_dtype,
        )

        outputs = encoder(inputs)

        self.assertEqual(
            outputs.shape,
            (2, 4, 16),
        )

        self.assertEqual(
            len(encoder.trainable_weights),
            7,
        )

        for layer in encoder._flatten_layers(
            include_self=True,
            recursive=True,
        ):
            self.assertEqual(
                layer.compute_dtype,
                compute_dtype,
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
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 1, 1, 1],
        ]

        self.assertAllClose(
            mask,
            expected,
        )
