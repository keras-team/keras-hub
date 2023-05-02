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
"""Random Sampler."""

import tensorflow as tf

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.samplers.sampler import Sampler
from keras_nlp.samplers.sampler import call_args_docstring
from keras_nlp.utils.python_utils import format_docstring


@format_docstring(call_args=call_args_docstring)
@keras_nlp_export("keras_nlp.samplers.RandomSampler")
class RandomSampler(Sampler):
    """Random Sampler class.

    This sampler implements random sampling. Briefly, random sampler randomly
    selects a token from the entire distribution of the tokens, with selection
    chance determined by the probability of each token.

    Args:
        seed: int, defaults to None. The random seed.

    Call arguments:
        {{call_args}}

    Examples:
    ```python
    # Use a simple alphabet of lowercase characters with ids in range [0, 25].
    int_lookup = {i: chr(i + ord('a')) for i in range(26)}
    char_lookup = {v: k for k, v in int_lookup.items()}
    batch_size, length, vocab_size = 1, 12, len(int_lookup)

    def next(prompt, state, index):
        hidden_states = tf.ones((batch_size, 10))
        # A uniform distribution over our alphabet.
        logits = tf.ones((batch_size, vocab_size))
        return logits, hidden_states, state

    output = keras_nlp.samplers.RandomSampler()(
        next=next,
        prompt=tf.fill((batch_size, length,), char_lookup['z']),
        index=5,
    )
    print(["".join([int_lookup[i] for i in s]) for s in output.numpy()])
    # >>> ['zzzzzcpnjqij']
    ```
    """

    def __init__(
        self,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.seed = seed

    def get_next_token(self, probabilities):
        # Sample the next token from the probability distribution.
        next_token_id = tf.random.categorical(
            tf.math.log(probabilities), 1, seed=self.seed, dtype="int32"
        )
        return tf.squeeze(next_token_id, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "seed": self.seed,
            }
        )
        return config
