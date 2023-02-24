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

    def test_cache_call_is_correct(self):
        batch_size = 2
        seq_len = 5
        num_heads = 2
        key_dim = 4

        layer = CachedMultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        x = tf.random.uniform(shape=[batch_size, seq_len, num_heads * key_dim])
        cache = tf.zeros([batch_size, 2, seq_len, num_heads, key_dim])
        # Build the intial cache.
        intial_seq_len = 2
        initial_inputs = x[:, :intial_seq_len, :]
        outputs = tf.zeros_like(x)
        output, cache = layer(
            query=initial_inputs,
            value=initial_inputs,
            use_causal_mask=True,
            cache=cache,
        )
        # Update the outputs in place.
        outputs = dynamic_update_slice(outputs, output, [0, 0, 0])

        def call(i, cache, outputs):
            def loop_body(i, cache, outputs):
                # Compute the rest tokens.
                current_input = x[:, i : i + 1, :]
                output, cache = layer(
                    query=current_input,
                    value=current_input,
                    use_causal_mask=True,
                    cache=cache,
                    cache_index=i,
                )
                outputs = dynamic_update_slice(outputs, output, [0, i, 0])
                return i + 1, cache, outputs

            i, cache, cached_outputs = tf.while_loop(
                cond=lambda i, cache, outputs: i < seq_len,
                body=loop_body,
                loop_vars=[i, cache, outputs],
            )
            return cached_outputs

        cached_outputs = call(intial_seq_len, cache, outputs)
        graph_call = tf.function(call)
        graph_cached_outputs = graph_call(intial_seq_len, cache, outputs)
        normal_outputs, _ = layer(query=x, value=x, use_causal_mask=True)
        self.assertAllClose(cached_outputs, normal_outputs)
        self.assertAllClose(graph_cached_outputs, normal_outputs)
