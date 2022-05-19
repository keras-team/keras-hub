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

"""Text generation utilities."""

import tensorflow as tf


def greedy_search(
    token_probability_fn,
    prompt,
    max_length,
    end_token_id=None,
    pad_token_id=0,
):
    """
    Text generation utility based on greedy search.

    Greedy search always appends the token having the largest probability to
    existing sequence.

    Args:
        token_probability_fn: a callable, which takes in input_sequence
            and output the probability distribution of the next token.
        prompt: a list or a Tensor, can be 1D or 2D, the initial tokens to
            append generated tokens.
        max_length: int. The max length of generated text.
        end_token_id: int, defaults to None. The token marking the end of the
            sequence, once encountered the generation is finished for the exact
            sequence. If None, every sequence is generated up to `max_length`.
            If set, all tokens after encountering `end_token_id` will be
            replaced with `pad_token_id`.
        pad_token_id: int, defaults to 0. The pad token after `end_token_id`
            is received.

    Returns:
        A 1D int Tensor, or 2D int RaggedTensor representing the generated
        sequences.

    Examples:
    ```python
    VOCAB_SIZE = 10
    FEATURE_SIZE = 16

    # Create a dummy model to predict the next token.
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=[None]),
            tf.keras.layers.Embedding(
                input_dim=VOCAB_SIZE,
                output_dim=FEATURE_SIZE,
            ),
            tf.keras.layers.Dense(VOCAB_SIZE, activation="softmax"),
        ]
    )

    # Define a function that outputs the next token's probability given the
    # input sequence.
    def token_probability_fn(inputs):
        return model(inputs)[:, -1, :]

    prompt = tf.random.uniform(shape=[5, 5], maxval=VOCAB_SIZE, dtype=tf.int64)

    # Print the generated sequence (token ids).
    keras_nlp.utils.greedy_search(
        token_probability_fn,
        prompt,
        max_length=10,
        end_token_id=0,)
    ```

    """
    if not tf.executing_eagerly():
        raise RuntimeError(
            "`keras_nlp.utils.greedy_search` currently requires an eager "
            "execution context. Please call `greedy_search` outside "
            "tf.function or run `tf.config.run_functions_eagerly(True)` to run "
            "tf.function in eager mode."
        )
    if isinstance(prompt, tf.RaggedTensor):
        raise ValueError(
            "RaggedTensor `prompt` is not supported, please "
            "provide `prompt` as a list or Tensor."
        )
    if not isinstance(prompt, tf.Tensor):
        prompt = tf.convert_to_tensor(prompt)
    input_is_1d = prompt.shape.rank == 1
    if input_is_1d:
        prompt = prompt[tf.newaxis, :]

    i = prompt.shape[1]
    while i < max_length:
        # If the prompt has reached our desired length, exit while loop.
        pred = token_probability_fn(prompt)
        next_token = tf.cast(tf.argmax(pred, axis=-1), dtype=prompt.dtype)
        # Append the next token to current sequence.
        prompt = tf.concat([prompt, next_token[:, tf.newaxis]], axis=-1)
        i += 1

    if end_token_id is not None:
        # Mask out tokens after `end_token_id` is encountered.
        # Find index of first end_token_id.
        end_indices = tf.math.argmax(prompt == end_token_id, -1)
        # Use max_length if no `end_token_id` is found.
        end_indices = tf.where(end_indices == 0, max_length, end_indices)
        # Build a mask including end_token and replace tokens after end_token
        # with `pad_token_id`.
        valid_indices = tf.sequence_mask(end_indices + 1, maxlen=max_length)
        prompt = tf.where(valid_indices, prompt, pad_token_id)

    if input_is_1d:
        return tf.squeeze(prompt)
    return prompt
