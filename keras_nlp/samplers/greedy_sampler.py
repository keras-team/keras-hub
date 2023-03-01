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
"""Greedy Sampler."""

import tensorflow as tf

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.samplers.sampler import Sampler
from keras_nlp.samplers.sampler import call_args_docstring
from keras_nlp.utils.python_utils import format_docstring


@format_docstring(call_args=call_args_docstring)
@keras_nlp_export("keras_nlp.samplers.GreedySampler")
class GreedySampler(Sampler):
    """Greedy sampler class.

    This sampler is implemented on greedy search, i.e., always picking up the
    token of the largest probability as the next token.

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

    sampler = keras_nlp.samplers.GreedySampler()
    # Print the generated sequence (token ids).
    print(sampler(prompt, token_probability_fn, max_length=10))
    ```
    """

    def __init__(self):
        super().__init__()

    def get_next_token(self, probs):
        return tf.argmax(probs, axis=-1)
