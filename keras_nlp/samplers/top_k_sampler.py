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
"""Top-k Sampler."""

import tensorflow as tf

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.samplers.sampler import Sampler
from keras_nlp.samplers.sampler import call_args_docstring
from keras_nlp.utils.python_utils import format_docstring


@format_docstring(call_args=call_args_docstring)
@keras_nlp_export("keras_nlp.samplers.TopKSampler")
class TopKSampler(Sampler):
    """Top-K Sampler class.

    This sampler implements top-k search algorithm. Briefly, top-k algorithm
    randomly selects a token from the tokens of top K probability, with
    selection chance determined by the probability.

    Args:
        k: int, the `k` value of top-k.
        seed: int, defaults to None. The random seed.

    Call arguments:
        {{call_args}}

    Examples:
    ```python
    # Use a simple alphabet of lowercase characters with ids in range [0, 25].
    int_lookup = {i: chr(i + ord('a')) for i in range(26)}
    char_lookup = {v: k for k, v in int_lookup.items()}
    batch_size, length, vocab_size = 1, 12, len(int_lookup)

    def next(prompt, cache, index):
        hidden_states = tf.ones((batch_size, 10))
        # A uniform distribution over our alphabet.
        logits = tf.ones((batch_size, vocab_size))
        return logits, hidden_states, cache

    output = keras_nlp.samplers.TopKSampler(k=3)(
        next=next,
        prompt=tf.fill((batch_size, length,), char_lookup['z']),
        index=5,
    )
    print(["".join([int_lookup[i] for i in s]) for s in output.numpy()])
    # >>> ['zzzzzacbbcaa']
    ```
    """

    def __init__(
        self,
        k=5,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.k = k
        self.seed = seed

    def get_next_token(self, probabilities):
        # Filter out top-k tokens.
        top_k_pred, top_k_indices = tf.math.top_k(
            probabilities, k=self.k, sorted=False
        )
        # Sample the next token from the probability distribution.
        next_token = tf.random.categorical(
            tf.math.log(top_k_pred), 1, seed=self.seed
        )

        # Rearrange to get the next token idx from the original order.
        return tf.gather_nd(top_k_indices, next_token, batch_dims=1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "k": self.k,
                "seed": self.seed,
            }
        )
        return config
