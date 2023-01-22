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
from tensorflow import keras

from keras_nlp.samplers.sampler import Sampler
from keras_nlp.samplers.sampler import base_sampler_args_docstring
from keras_nlp.samplers.sampler import call_args_docstring
from keras_nlp.utils.python_utils import format_docstring


@format_docstring(
    base_sampler_args=base_sampler_args_docstring, call_args=call_args_docstring
)
@keras.utils.register_keras_serializable(package="keras_nlp")
class TopKSampler(Sampler):
    """Top-K Sampler class.

    This sampler implements top-k search algorithm. Briefly top-k algorithm
    randomly selects a token from the tokens of top K probability, with
    selection chance determined by the probability.

    Args:
        k: int, the `k` value of top-k.
        seed: int, defaults to None. The random seed.
        {{base_sampler_args}}

    Call Args:
        {{call_args}}

    Examples:
    ```python
    VOCAB_SIZE = 10

    # Create a dummy model to predict the next token.
    model = keras.Sequential(
        [
            keras.Input(shape=[None]),
            keras.layers.Embedding(
                input_dim=VOCAB_SIZE,
                output_dim=16,
            ),
            keras.layers.Dense(VOCAB_SIZE, activation="softmax"),
        ]
    )

    # Define a function that outputs the next token's probability for each token
    # in the input sequence.
    def token_probability_fn(inputs, mask):
        return model(inputs)

    prompt = tf.fill((8, 1), 1)

    sampler = keras_nlp.samplers.TopKSampler(k=5)
    # Print the generated sequence (token ids).
    print(sampler(prompt, token_probability_fn, max_length=10))
    ```
    """

    def __init__(
        self,
        k=5,
        seed=None,
        jit_compile=True,
        run_eagerly=False,
    ):
        self.k = k
        self.seed = seed
        super().__init__(jit_compile=jit_compile, run_eagerly=run_eagerly)

    def get_next_token(self, next_token_probs):
        # Filter out top-k tokens.
        top_k_pred, top_k_indices = tf.math.top_k(
            next_token_probs, k=self.k, sorted=False
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
