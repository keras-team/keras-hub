from keras import ops
from keras import random

from keras_hub.src.layers.modeling.cached_multi_head_attention import (
    CachedMultiHeadAttention,
)
from keras_hub.src.tests.test_case import TestCase


class CachedMultiHeadAttentionTest(TestCase):
    def test_layer_behaviors(self):
        self.run_layer_test(
            cls=CachedMultiHeadAttention,
            init_kwargs={
                "num_heads": 2,
                "key_dim": 4,
                "dropout": 0.1,
            },
            input_data={
                "query": random.uniform(shape=(2, 4, 6)),
                "value": random.uniform(shape=(2, 4, 6)),
            },
            expected_output_shape=(2, 4, 6),
            expected_num_trainable_weights=8,
            expected_num_non_trainable_variables=1,
        )

    def test_cache_call_is_correct(self):
        batch_size = 2
        seq_len = 5
        num_heads = 2
        key_dim = 4
        hidden_dim = num_heads * key_dim

        input_shape = (batch_size, seq_len, hidden_dim)
        x = random.uniform(shape=input_shape)
        input_cache = ops.zeros((batch_size, 2, seq_len, num_heads, key_dim))
        # Use a causal mask.
        mask = ops.tril(ops.ones((seq_len, seq_len)))
        outputs = ops.zeros_like(x)

        layer = CachedMultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        no_loop_outputs, no_loop_cache = layer(
            x,
            x,
            cache=input_cache,
            cache_update_index=0,
            attention_mask=mask,
        )

        def loop_body(i, outputs, cache):
            # Compute the rest tokens.
            next_input = ops.slice(x, (0, i, 0), (batch_size, 1, hidden_dim))
            next_mask = ops.slice(mask, (i, 0), (1, seq_len))
            next_output, cache = layer(
                query=next_input,
                value=next_input,
                cache=cache,
                cache_update_index=i,
                attention_mask=next_mask,
            )
            outputs = ops.slice_update(outputs, [0, i, 0], next_output)
            return i + 1, outputs, cache

        def call(outputs, cache):
            _, outputs, cache = ops.while_loop(
                cond=lambda i, outputs, cache: i < seq_len,
                body=loop_body,
                loop_vars=[0, outputs, cache],
            )
            return outputs, cache

        output, output_cache = call(outputs, input_cache)

        self.assertAllClose(output, no_loop_outputs)
        self.assertAllClose(output_cache, no_loop_cache)

    def test_training_propagation(self):
        batch_size = 2
        seq_len = 5
        num_heads = 2
        key_dim = 4
        hidden_dim = num_heads * key_dim

        input_shape = (batch_size, seq_len, hidden_dim)
        x = random.uniform(shape=input_shape)

        layer = CachedMultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=0.99999,  # Zeros out the outputs after the dropout layer
        )
        outputs = layer(x, x, training=True)

        # Custom computation with dropout rate sets to about 1.0
        value = layer._value_dense(x)
        attention_scores = ops.zeros((batch_size, num_heads, seq_len, seq_len))
        attention_output = ops.einsum(
            layer._combine_equation, attention_scores, value
        )
        attention_output = layer._output_dense(attention_output)

        self.assertAllClose(outputs, attention_output, atol=1e-5)

    def test_returns_attention_scores(self):
        batch_size = 2
        seq_len = 4
        num_heads = 2
        key_dim = 4
        hidden_dim = num_heads * key_dim

        query = random.uniform(shape=(batch_size, seq_len, hidden_dim))
        value = random.uniform(shape=(batch_size, seq_len, hidden_dim))

        layer = CachedMultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        output, scores = layer(query, value, return_attention_scores=True)

        self.assertEqual(output.shape, (batch_size, seq_len, hidden_dim))
        self.assertIsNotNone(scores)
        self.assertEqual(scores.shape[0], batch_size)
        self.assertEqual(len(scores.shape), 4)  # Expected: (B, H, T, S)