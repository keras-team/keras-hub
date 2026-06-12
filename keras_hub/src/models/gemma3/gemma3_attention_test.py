from unittest.mock import patch

from absl.testing import parameterized
from keras import ops

from keras_hub.src.models.gemma3.gemma3_attention import CachedGemma3Attention
from keras_hub.src.tests.test_case import TestCase


class CachedGemma3AttentionTest(TestCase, parameterized.TestCase):
    def setUp(self):
        self.head_dim = 64
        self.num_query_heads = 4
        self.num_key_value_heads = 2
        self.hidden_dim = 128
        self.batch_size = 2
        self.seq_len = 16

    @parameterized.named_parameters(
        ("standard_attention", False),
        ("fused_attention", True),
    )
    def test_gqa_forward_pass(self, use_fused_attention):
        """Tests that GQA executes correctly without shape mismatches."""
        layer = CachedGemma3Attention(
            head_dim=self.head_dim,
            num_query_heads=self.num_query_heads,
            num_key_value_heads=self.num_key_value_heads,
        )
        layer.build((self.batch_size, self.seq_len, self.hidden_dim))

        x = ops.ones(
            (self.batch_size, self.seq_len, self.hidden_dim), dtype="float32"
        )
        attention_mask = ops.ones(
            (self.batch_size, self.seq_len, self.seq_len), dtype="int32"
        )

        with patch.object(
            layer, "_use_fused_attention_op", return_value=use_fused_attention
        ):
            output = layer(x, attention_mask=attention_mask)

            self.assertEqual(
                output.shape, (self.batch_size, self.seq_len, self.hidden_dim)
            )

    @parameterized.named_parameters(
        ("standard_attention_cache", False),
        ("fused_attention_cache", True),
    )
    def test_gqa_cached_generation(self, use_fused_attention):
        """Tests GQA execution with key-value cache during generation."""
        layer = CachedGemma3Attention(
            head_dim=self.head_dim,
            num_query_heads=self.num_query_heads,
            num_key_value_heads=self.num_key_value_heads,
        )
        layer.build((self.batch_size, self.seq_len, self.hidden_dim))

        # Input is a single newly generated token
        x = ops.ones((self.batch_size, 1, self.hidden_dim), dtype="float32")
        attention_mask = ops.ones(
            (self.batch_size, 1, self.seq_len), dtype="int32"
        )

        cache = ops.zeros(
            (
                self.batch_size,
                2,
                self.seq_len,
                self.num_key_value_heads,
                self.head_dim,
            ),
            dtype="float32",
        )

        with patch.object(
            layer, "_use_fused_attention_op", return_value=use_fused_attention
        ):
            output, updated_cache = layer(
                x=x,
                attention_mask=attention_mask,
                cache=cache,
                cache_update_index=5,
            )

            self.assertEqual(
                output.shape, (self.batch_size, 1, self.hidden_dim)
            )
            self.assertEqual(
                updated_cache.shape,
                (
                    self.batch_size,
                    2,
                    self.seq_len,
                    self.num_key_value_heads,
                    self.head_dim,
                ),
            )
