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
"""Greedy Sampler."""

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
class Greedy(Sampler):
    """Greedy sampler class.

    This sampler is implemented on greedy search, i.e., always picking up the
    token of the largest probability as the next token.

    Args:
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

    sampler = keras_nlp.samplers.Greedy()
    # Print the generated sequence (token ids).
    print(sampler(prompt, token_probability_fn, 10))
    ```
    """

    def __init__(
        self,
        jit_compile=True,
    ):
        super().__init__(jit_compile)

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
        # The index of the last non-padding token in prompt. Since all sequences
        # are aligned to the right side, the index is the same for all.
        current_index = max_length - num_steps

        def one_step(current_index, prompt, mask):
            probs = token_probability_fn(prompt, mask)
            next_token_prob = tf.gather(
                probs,
                tf.repeat(current_index - 1, batch_size),
                axis=1,
                batch_dims=1,
            )
            next_token = tf.cast(
                tf.argmax(next_token_prob, axis=-1), dtype=prompt.dtype
            )
            next_token = tf.where(
                mask[:, current_index], prompt[:, current_index], next_token
            )
            mask = tf.tensor_scatter_nd_update(
                tensor=mask,
                indices=tf.stack(
                    (
                        tf.cast(
                            tf.range(batch_size), dtype=current_index.dtype
                        ),
                        tf.repeat(current_index, batch_size),
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
                        tf.cast(
                            tf.range(batch_size), dtype=current_index.dtype
                        ),
                        tf.repeat(current_index, batch_size),
                    ),
                    axis=1,
                ),
                updates=next_token,
            )

            current_index = tf.add(current_index, 1)
            return (current_index, prompt, mask)

        # Run a while loop till `max_length` of tokens has been generated.
        current_index, prompt, mask = tf.while_loop(
            cond=lambda current_index, prompt, mask: tf.less(
                current_index, max_length
            ),
            body=one_step,
            loop_vars=(current_index, prompt, mask),
        )
        return prompt
