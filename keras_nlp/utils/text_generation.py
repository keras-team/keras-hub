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


def _handle_end_token(next_token, end_token_received, end_token_id):
    filtered_next_token = next_token * (1 - end_token_received)
    end_token_received = tf.cast(
        filtered_next_token == end_token_id, dtype=next_token.dtype
    )
    return filtered_next_token, end_token_received


def generate_text_greedy(
    get_next_token_probability_fn, seed, max_length, end_token_id=None
):
    """
    Text generation utility based on greedy search.

    Greedy search always appends the token having the largest probability to
    existing sequence.

    Args:
        get_next_token_probability_fn: a callable, which takes in input_sequence
            and output the probability distribution of the next token.
        seed: a list, the initial tokens to append generated tokens.
        max_length: int. The max length of generated text.
        end_token_id: int, defaults to None. The token marking the end of the
            sequence, once encountered the generation is finished for the exact
            sequence. If None, every sequence is generated to `max_length`.

    Returns:
        A 1D int Tensor, or 2D int Tensor of shape (batch_size, `max_length`)
        representing the generated sequences.

    Examples:
    ```python
    VOCAB_SIZE = 10
    FEATURE_SIZE = 16

    # Create a dummy model to predict the next token.
    model = tf.keras.Sequential([
        tf.keras.Input(shape=[None]),
        tf.keras.layers.Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=FEATURE_SIZE,),
        tf.keras.layers.Dense(VOCAB_SIZE),
        tf.keras.layers.Softmax(),
    ])

    # Define a function that outputs the next token's probability given the
    # input sequence.
    def get_next_token_probability_fn(inputs):
        return model(inputs)[:, -1, :]

    inputs = tf.random.uniform(shape=[5, 5], maxval=VOCAB_SIZE, dtype=tf.int64)

    # Print the generated sequence (token ids).
    print(generate_text_greedy(
        get_next_token_probability_fn,
        inputs,
        max_length=10,
        end_token=0,))
    ```

    """
    if 0 in seed.shape:
        raise ValueError("Seed must not be empty, but received empty seed.")
    input_is_1d = seed.shape.rank == 1
    if input_is_1d:
        seed = seed[tf.newaxis, :]

    # Store if the end token has been generated for each sequence.
    end_token_received = None
    if end_token_id is not None:
        end_token_received = tf.cast(
            (seed[:, -1] == end_token_id), dtype=seed.dtype
        )

    def helper(seed, end_token_received):
        if seed.shape[1] >= max_length:
            # If the seed has reached our desired length, exit recursion.
            return seed
        pred = get_next_token_probability_fn(seed)
        next_token = tf.cast(tf.argmax(pred, axis=-1), dtype=seed.dtype)
        if end_token_id is not None:
            # Replace the next token with `end_token_id` if end token has
            # appeared in the sequenece.
            next_token, end_token_received = _handle_end_token(
                next_token, end_token_received, end_token_id
            )
        # Append the next token to current sequence.
        seed = tf.concat([seed, next_token[:, tf.newaxis]], axis=-1)
        return helper(seed, end_token_received)

    generated_sequence = helper(seed, end_token_received)
    if input_is_1d:
        return tf.squeeze(generated_sequence)
    return generated_sequence
