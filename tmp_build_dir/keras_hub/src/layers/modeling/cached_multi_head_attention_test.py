from keras import ops
from keras import random

from keras_hub.src.layers.modeling.cached_multi_head_attention import (
    CachedMultiHeadAttention,
)
from keras_hub.src.tests.test_case import TestCase


class CachedMultiHeadAttentionTest(TestCase):
    def setUp(self):
        super().setUp()
        self.batch_size = 2
        self.seq_len = 5
        self.num_heads = 2
        self.key_dim = 4
        self.hidden_dim = self.num_heads * self.key_dim

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
        input_shape = (self.batch_size, self.seq_len, self.hidden_dim)
        x = random.uniform(shape=input_shape)
        input_cache = ops.zeros(
            (
                self.batch_size,
                2,
                self.seq_len,
                self.num_heads,
                self.key_dim,
            )
        )
        # Use a causal mask.
        mask = ops.tril(ops.ones((self.seq_len, self.seq_len)))
        outputs = ops.zeros_like(x)

        layer = CachedMultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
        )
        no_loop_outputs, no_loop_cache = layer(
            x,
            x,
            cache=input_cache,
            cache_update_index=0,
            attention_mask=mask,
        )

        def loop_body(i, outputs, cache):
            # Compute the rest tokens.
            next_input = ops.slice(
                x,
                (0, i, 0),
                (self.batch_size, 1, self.hidden_dim),
            )
            next_mask = ops.slice(mask, (i, 0), (1, self.seq_len))
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
                cond=lambda i, outputs, cache: i < self.seq_len,
                body=loop_body,
                loop_vars=[0, outputs, cache],
            )
            return outputs, cache

        output, output_cache = call(outputs, input_cache)

        self.assertAllClose(output, no_loop_outputs)
        self.assertAllClose(output_cache, no_loop_cache)

    def test_return_attention_scores(self):
        x = random.uniform(
            shape=(self.batch_size, self.seq_len, self.hidden_dim)
        )

        layer = CachedMultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
        )
        output, attention_scores = layer(
            x,
            x,
            return_attention_scores=True,
        )

        self.assertIsNotNone(attention_scores)
        self.assertAllEqual(
            output.shape,
            [self.batch_size, self.seq_len, self.hidden_dim],
        )
        self.assertAllEqual(
            attention_scores.shape,
            [
                self.batch_size,
                self.num_heads,
                self.seq_len,
                self.seq_len,
            ],
        )
        score_sums = ops.sum(attention_scores, axis=-1)
        self.assertAllClose(
            score_sums,
            ops.ones_like(score_sums),
            atol=1e-5,
        )

    def test_return_attention_scores_with_cache(self):
        x = random.uniform(
            shape=(self.batch_size, self.seq_len, self.hidden_dim)
        )
        input_cache = ops.zeros(
            (
                self.batch_size,
                2,
                self.seq_len,
                self.num_heads,
                self.key_dim,
            )
        )
        mask = ops.tril(ops.ones((self.seq_len, self.seq_len)))

        layer = CachedMultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
        )
        output, attention_scores, output_cache = layer(
            x,
            x,
            cache=input_cache,
            cache_update_index=0,
            attention_mask=mask,
            return_attention_scores=True,
        )

        self.assertIsNotNone(attention_scores)
        self.assertAllEqual(
            output.shape,
            [self.batch_size, self.seq_len, self.hidden_dim],
        )
        self.assertAllEqual(
            attention_scores.shape,
            [
                self.batch_size,
                self.num_heads,
                self.seq_len,
                self.seq_len,
            ],
        )
        self.assertAllEqual(output_cache.shape, input_cache.shape)

    def test_training_propagation(self):
        input_shape = (self.batch_size, self.seq_len, self.hidden_dim)
        x = random.uniform(shape=input_shape)

        layer = CachedMultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            dropout=0.99999,  # Zeros out the outputs after the dropout layer
        )
        outputs = layer(x, x, training=True)

        # Custom computation with dropout rate sets to about 1.0
        value = layer._value_dense(x)
        attention_scores = ops.zeros(
            (
                self.batch_size,
                self.num_heads,
                self.seq_len,
                self.seq_len,
            )
        )
        attention_output = ops.einsum(
            layer._combine_equation, attention_scores, value
        )
        attention_output = layer._output_dense(attention_output)

        self.assertAllClose(outputs, attention_output, atol=1e-5)
