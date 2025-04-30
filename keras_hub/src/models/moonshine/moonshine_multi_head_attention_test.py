import keras

from keras_hub.src.models.moonshine.moonshine_multi_head_attention import (
    MoonshineMultiHeadAttention,
)
from keras_hub.src.tests.test_case import TestCase


class MoonshineMultiHeadAttentionTest(TestCase):
    def setUp(self):
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
            "cache_mode": "none",
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
        self.value = self.key  # For testing purposes
        self.rotary_embedding = keras.random.normal(
            (self.query_seq_len, self.rotary_dim)
        )
        self.attention_mask = keras.ops.ones(
            (self.batch_size, self.key_seq_len), dtype="bool"
        )

    def test_initialization(self):
        self.assertEqual(self.attention_layer.num_heads, self.num_heads)
        self.assertEqual(self.attention_layer.key_dim, self.key_dim)
        self.assertFalse(self.attention_layer.attention_bias)
        self.assertTrue(self.attention_layer.apply_rotary_embedding)

    def test_forward_pass_without_caching(self):
        self.attention_layer.apply_rotary_embedding = (
            False  # Test cross-attention
        )
        output = self.attention_layer(
            query=self.query,
            key=self.key,
            value=self.value,
            rotary_embedding=self.rotary_embedding,
            attention_mask=self.attention_mask,
        )
        self.assertEqual(
            output.shape, (self.batch_size, self.query_seq_len, self.hidden_dim)
        )

    def test_precomputed_caching(self):
        self.attention_layer.build(
            query_shape=(self.batch_size, self.query_seq_len, self.hidden_dim),
            value_shape=(self.batch_size, self.key_seq_len, self.hidden_dim),
            key_shape=(self.batch_size, self.key_seq_len, self.hidden_dim),
        )
        self.attention_layer.cache_mode = "precomputed"
        self.attention_layer.apply_rotary_embedding = False
        key_proj = self.attention_layer._key_dense(self.key)
        value_proj = self.attention_layer._value_dense(self.value)
        output_precomputed = self.attention_layer(
            query=self.query,
            key=None,
            value=None,
            key_cache=key_proj,
            value_cache=value_proj,
            rotary_embedding=self.rotary_embedding,
            attention_mask=self.attention_mask,
        )
        self.attention_layer.cache_mode = "none"
        output_normal = self.attention_layer(
            query=self.query,
            key=self.key,
            value=self.value,
            rotary_embedding=self.rotary_embedding,
            attention_mask=self.attention_mask,
        )
        self.assertEqual(
            output_precomputed.shape,
            (self.batch_size, self.query_seq_len, self.hidden_dim),
        )
        self.assertAllClose(output_precomputed, output_normal, atol=1e-5)

    def test_autoregressive_caching(self):
        self.attention_layer.cache_mode = "autoregressive"
        self.attention_layer.use_causal_mask = True  # Ensure causal attention
        cache_k, cache_v = None, None
        outputs_auto = []
        for i in range(self.query_seq_len):
            query_i = self.query[:, i : i + 1, :]
            key_i = self.query[:, i : i + 1, :]  # Self-attention
            value_i = self.query[:, i : i + 1, :]
            rotary_i = self.rotary_embedding[i : i + 1, :]
            output_i, new_cache_k, new_cache_v = self.attention_layer(
                query=query_i,
                key=key_i,
                value=value_i,
                rotary_embedding=rotary_i,
                key_cache=cache_k,
                value_cache=cache_v,
            )
            outputs_auto.append(output_i)
            self.assertEqual(
                output_i.shape, (self.batch_size, 1, self.hidden_dim)
            )
            self.assertEqual(
                new_cache_k.shape,
                (self.batch_size, i + 1, self.num_heads, self.key_dim),
            )
            self.assertEqual(
                new_cache_v.shape,
                (self.batch_size, i + 1, self.num_heads, self.key_dim),
            )
            cache_k, cache_v = new_cache_k, new_cache_v
        outputs_auto = keras.ops.concatenate(outputs_auto, axis=1)
        self.attention_layer.cache_mode = "none"
        self.attention_layer.use_causal_mask = (
            True  # Consistent with autoregressive
        )
        output_full = self.attention_layer(
            query=self.query,
            key=self.query,
            value=self.query,
            rotary_embedding=self.rotary_embedding,
        )
        self.assertAllClose(outputs_auto, output_full, atol=1e-5)

    def test_forward_pass_with_causal_mask(self):
        self.attention_layer.use_causal_mask = True
        output = self.attention_layer(
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
