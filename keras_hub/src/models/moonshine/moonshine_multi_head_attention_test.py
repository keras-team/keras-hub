import keras

from keras_hub.src.models.moonshine.moonshine_multi_head_attention import (
    MoonshineMultiHeadAttention,
)
from keras_hub.src.tests.test_case import TestCase


class MoonshineMultiHeadAttentionTest(TestCase):
    def setUp(self):
        super().setUp()
        self.num_heads = 4
        self.key_dim = 16
        self.hidden_dim = self.num_heads * self.key_dim
        self.init_kwargs = {
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "value_dim": None,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "use_causal_mask": False,
            "apply_rotary_embedding": True,
        }
        self.attention_layer = MoonshineMultiHeadAttention(**self.init_kwargs)
        self.batch_size = 2
        self.query_seq_len = 10
        self.key_seq_len = 16
        self.rotary_dim = int(
            self.key_dim * 0.62
        )  # Default partial_rotary_factor = 0.62
        self.rotary_dim = (self.rotary_dim // 2) * 2  # Ensure even
        self.rotary_dim = self.rotary_dim // 2  # Half for freqs, e.g., 4
        self.query = keras.random.normal(
            (self.batch_size, self.query_seq_len, self.hidden_dim)
        )
        self.key = keras.random.normal(
            (self.batch_size, self.key_seq_len, self.hidden_dim)
        )
        self.rotary_embedding = keras.random.normal(
            (self.query_seq_len, self.rotary_dim)
        )

    def test_initialization(self):
        self.assertEqual(self.attention_layer.num_heads, self.num_heads)
        self.assertEqual(self.attention_layer.key_dim, self.key_dim)
        self.assertFalse(self.attention_layer.attention_bias)
        self.assertTrue(self.attention_layer.apply_rotary_embedding)

    def test_forward_pass_with_causal_mask(self):
        self.attention_layer.use_causal_mask = True
        output, _ = self.attention_layer(
            query=self.query,
            key=self.query,  # Self-attention for causal test
            value=self.query,
            rotary_embedding=self.rotary_embedding,
        )
        self.assertEqual(
            output.shape, (self.batch_size, self.query_seq_len, self.hidden_dim)
        )

    def test_serialization(self):
        instance = MoonshineMultiHeadAttention(**self.init_kwargs)
        self.run_serialization_test(instance=instance)

    def test_forward_pass_with_dynamic_rotary_embedding(self):
        head_dim = self.key_dim  # 16
        max_position_embeddings = 5
        base_value = 10000
        partial_rotary_factor = 0.62
        rotary_dim = int(head_dim * partial_rotary_factor)  # 9
        rotary_dim = (rotary_dim // 2) * 2  # 8
        original_inv_freq = 1.0 / (
            base_value
            ** (
                keras.ops.arange(0, rotary_dim // 2, dtype="float32")
                / (rotary_dim // 2)
            )
        )
        seq_len = self.query_seq_len  # 10
        scaling = float(max_position_embeddings) / seq_len
        current_inv_freq = original_inv_freq * scaling
        position_ids = keras.ops.arange(seq_len, dtype="float32")
        freqs = position_ids[:, None] * current_inv_freq[None, :]

        self.key_seq_len = self.query_seq_len
        key = keras.random.normal(
            (self.batch_size, self.key_seq_len, self.hidden_dim)
        )

        output, _ = self.attention_layer(
            query=self.query,
            key=key,
            value=key,
            rotary_embedding=freqs,
        )
        self.assertEqual(
            output.shape, (self.batch_size, self.query_seq_len, self.hidden_dim)
        )
