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


def _handle_end_token_id(next_token, end_token_id_received, end_token_id):
    filtered_next_token = next_token * (1 - end_token_id_received)
    end_token_id_received = tf.cast(
        filtered_next_token == end_token_id, dtype=next_token.dtype
    )
    return filtered_next_token, end_token_id_received


def greedy_search(token_probability_fn, prompt, max_length, end_token_id=None):
    """
    Text generation utility based on greedy search.

    Greedy search always appends the token having the largest probability to
    existing sequence.

    Args:
        token_probability_fn: a callable, which takes in input_sequence
            and output the probability distribution of the next token.
        prompt: a list, the initial tokens to append generated tokens.
        max_length: int. The max length of generated text.
        end_token_id: int, defaults to None. The token marking the end of the
            sequence, once encountered the generation is finished for the exact
            sequence. If None, every sequence is generated up to `max_length`.

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
    keras_nlp.greedy_search(
        token_probability_fn,
        prompt,
        max_length=10,
        end_token_id=0,)
    ```

    """
    if 0 in prompt.shape:
        raise ValueError("prompt must not be empty, but received empty prompt.")
    input_is_1d = prompt.shape.rank == 1
    if input_is_1d:
        prompt = prompt[tf.newaxis, :]

    # Store if the end token has been generated for each sequence.
    end_token_id_received = None
    if end_token_id is not None:
        end_token_id_received = tf.cast(
            (prompt[:, -1] == end_token_id), dtype=prompt.dtype
        )

    def get_subsequent_tokens(prompt, end_token_id_received):
        if prompt.shape[1] >= max_length:
            # If the prompt has reached our desired length, exit recursion.
            return prompt
        pred = token_probability_fn(prompt)
        next_token = tf.cast(tf.argmax(pred, axis=-1), dtype=prompt.dtype)
        if end_token_id is not None:
            # Replace the next token with `end_token_id` if end token has
            # appeared in the sequenece.
            next_token, end_token_id_received = _handle_end_token_id(
                next_token, end_token_id_received, end_token_id
            )
        # Append the next token to current sequence.
        prompt = tf.concat([prompt, next_token[:, tf.newaxis]], axis=-1)
        return get_subsequent_tokens(prompt, end_token_id_received)

    generated_sequence = get_subsequent_tokens(prompt, end_token_id_received)
    if input_is_1d:
        return tf.squeeze(generated_sequence)
    return generated_sequence
