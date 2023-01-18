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
"""Top-p Sampler."""

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
class TopPSampler(Sampler):
    """Top-P Sampler class.

    This sampler implements top-p search algorithm. Top-p search selects tokens
    from the smallest subset of output probabilities that sum to greater than
    `p`. Put in another way, top-p will first order token predictions by
    likelihood, and ignore all tokens after the cumulative probability of
    selected tokens exceeds `p`, then select a token from the remaining tokens.

    Args:
        p: float, the `p` value of top-p.
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

    sampler = keras_nlp.samplers.TopPSampler(p=0.1)
    # Print the generated sequence (token ids).
    print(sampler(prompt, token_probability_fn, max_length=10))
    ```
    """

    def __init__(
        self,
        p=0.1,
        seed=None,
        jit_compile=True,
        run_eagerly=False,
    ):
        self.p = p
        self.seed = seed
        super().__init__(jit_compile=jit_compile, run_eagerly=run_eagerly)

    def get_next_token(self, next_token_probs):
        # Sort preds in descending order.
        sorted_preds, sorted_indices = tf.math.top_k(
            next_token_probs, k=tf.shape(next_token_probs)[1], sorted=True
        )
        # Calculate cumulative probability distribution.
        cumulative_probs = tf.math.cumsum(sorted_preds, axis=-1)
        # Create a mask for the tokens to keep.
        keep_mask = cumulative_probs <= self.p
        # Shift to include the last token that exceed p.
        shifted_keep_mask = tf.concat(
            [tf.ones_like(keep_mask[:, :1]), keep_mask[:, :-1]], axis=-1
        )
        # Filter out unmasked tokens and sample from filtered distribution.
        probs = tf.where(
            shifted_keep_mask,
            sorted_preds,
            tf.zeros(tf.shape(next_token_probs), dtype=sorted_preds.dtype),
        )
        sorted_next_token = tf.random.categorical(
            tf.math.log(probs), 1, seed=self.seed
        )
        return tf.gather_nd(sorted_indices, sorted_next_token, batch_dims=1)

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "p": self.p,
                "seed": self.seed,
            }
        )
        return config
