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

from keras_nlp.backend import config
from keras_nlp.backend import ops
from keras_nlp.backend import random
from keras_nlp.layers.modeling.cached_multi_head_attention import (
    CachedMultiHeadAttention,
)
from keras_nlp.tests.test_case import TestCase


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
            # Keras 2 does not handle mixed precision correctly when not set
            # globally.
            run_precision_checks=config.keras_3(),
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
