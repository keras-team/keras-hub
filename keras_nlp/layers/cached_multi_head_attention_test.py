# Copyright 2023 The KerasNLP Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for CachedMultiHeadAttention."""

import tensorflow as tf
from absl.testing import parameterized
from tensorflow.compiler.tf2xla.python.xla import dynamic_update_slice

from keras_nlp.layers.cached_multi_head_attention import (
    CachedMultiHeadAttention,
)


class CachedMultiHeadAttentionTest(tf.test.TestCase, parameterized.TestCase):
    def test_valid_call(self):
        layer = CachedMultiHeadAttention(num_heads=2, key_dim=4)
        x = tf.random.uniform(shape=[2, 2, 8])
        layer(query=x, value=x)

    @parameterized.named_parameters(
        ("graph", False),
        ("eager", True),
    )
    def test_cache_call_is_correct(self, eager):
        batch_size = 2
        seq_len = 5
        num_heads = 2
        key_dim = 4

        layer = CachedMultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        dtype = layer.compute_dtype
        x = tf.random.uniform(
            shape=[batch_size, seq_len, num_heads * key_dim], dtype=dtype
        )
        cache = tf.zeros(
            [batch_size, 2, seq_len, num_heads, key_dim], dtype=dtype
        )
        # Use a causal mask.
        mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        outputs = tf.zeros_like(x)

        def call(outputs, cache):
            def loop_body(i, outputs, cache):
                # Compute the rest tokens.
                next_input = x[:, i : i + 1, :]
                next_mask = mask[i : i + 1, :]
                next_output, cache = layer(
                    query=next_input,
                    value=next_input,
                    cache=cache,
                    cache_index=i,
                    attention_mask=next_mask,
                )
                outputs = dynamic_update_slice(outputs, next_output, [0, i, 0])
                return i + 1, outputs, cache

            _, outputs, cache = tf.while_loop(
                cond=lambda i, outputs, cache: i < seq_len,
                body=loop_body,
                loop_vars=[0, outputs, cache],
            )
            return outputs, cache

        call = call if eager else tf.function(call)
        output, cache = call(outputs, cache)

        no_loop_outputs, _ = layer(x, x, attention_mask=mask)
        _, no_loop_cache = layer(x, x, cache=cache, attention_mask=mask)
        self.assertAllClose(output, no_loop_outputs)
        self.assertAllClose(cache, no_loop_cache)
