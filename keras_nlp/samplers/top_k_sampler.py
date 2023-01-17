# Copyright 2022 The KerasNLP Authors
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
from keras_nlp.samplers.sampler import sample_args_docstring
from keras_nlp.utils.python_utils import format_docstring


@format_docstring(
    base_sampler_args=base_sampler_args_docstring, call_args=call_args_docstring
)
@keras.utils.register_keras_serializable(package="keras_nlp")
class TopKSampler(Sampler):
    """Top-K Sampler class.

    This sampler implements the top-k search algorithm. Briefly top-k algorithm
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
    BATCH_SIZE = 8
    VOCAB_SIZE = 10
    FEATURE_SIZE = 16
    START_ID = 1

    # Create a dummy model to predict the next token.
    model = keras.Sequential(
        [
            keras.Input(shape=[None]),
            keras.layers.Embedding(
                input_dim=VOCAB_SIZE,
                output_dim=FEATURE_SIZE,
            ),
            keras.layers.Dense(VOCAB_SIZE, activation="softmax"),
        ]
    )

    # Define a function that outputs the next token's probability for each token
    # in the input sequence.
    def token_probability_fn(inputs, mask):
        return model(inputs)

    prompt = tf.fill((BATCH_SIZE, 1), START_ID)

    sampler = keras_nlp.samplers.TopKSampler(k=5)
    # Print the generated sequence (token ids).
    print(sampler(prompt, token_probability_fn, 10))
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
        super().__init__(jit_compile, run_eagerly)

    @format_docstring(sample_args=sample_args_docstring)
    def sample(
        self, prompt, token_probability_fn, mask, num_steps, from_logits=True
    ):
        """Sampling logic implementation.

        Args:
            {{sample_args}}
        """
        batch_size, max_length = tf.shape(prompt)[0], tf.shape(prompt)[1]
        max_length = tf.cast(max_length, num_steps.dtype)
        length = max_length - num_steps

        def one_step(length, prompt, mask):
            probs = token_probability_fn(prompt, mask)
            pred = tf.gather(
                probs, tf.repeat(length - 1, batch_size), axis=1, batch_dims=1
            )
            if from_logits:
                pred = keras.activations.softmax(pred, axis=-1)
            # Filter out top-k tokens.
            top_k_pred, top_k_indices = tf.math.top_k(
                pred, k=self.k, sorted=False
            )
            # Sample the next token from the probability distribution.
            next_token = tf.random.categorical(
                tf.math.log(top_k_pred), 1, seed=self.seed
            )

            # Rearrange to get the next token idx from the original order.
            next_token = tf.gather_nd(top_k_indices, next_token, batch_dims=1)
            next_token = tf.cast(next_token, dtype=prompt.dtype)
            next_token = tf.where(
                mask[:, length], prompt[:, length], next_token
            )

            mask = tf.tensor_scatter_nd_update(
                tensor=mask,
                indices=tf.stack(
                    (
                        tf.cast(tf.range(batch_size), dtype=length.dtype),
                        tf.repeat(length, batch_size),
                    ),
                    axis=1,
                ),
                updates=tf.repeat(True, batch_size),
            )

            # Append the next token to current sequence.
            prompt = tf.tensor_scatter_nd_update(
                tensor=prompt,
                indices=tf.stack(
                    (
                        tf.cast(tf.range(batch_size), dtype=length.dtype),
                        tf.repeat(length, batch_size),
                    ),
                    axis=1,
                ),
                updates=next_token,
            )

            length = tf.add(length, 1)
            return (length, prompt, mask)

        # Run a while loop till text of length `max_length` has been generated.
        length, prompt, mask = tf.while_loop(
            cond=lambda length, prompt, mask: tf.less(length, max_length),
            body=one_step,
            loop_vars=(length, prompt, mask),
        )

        return prompt
