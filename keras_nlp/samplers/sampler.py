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
"""Base sampler class."""

import tensorflow as tf


class Sampler:
    """Base sampler class.

    Args:
        {{base_optimizer_keyword_args}}

    Call Args:
        {{call_keyword_docstring}}

    Examples:
    ```python
    BATCH_SIZE = 8
    VOCAB_SIZE = 10
    FEATURE_SIZE = 16
    START_ID = 1
    END_ID = 2

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
    def token_probability_fn(inputs):
        return model(inputs)

    prompt = tf.fill((BATCH_SIZE, 1), START_ID)

    sampler = keras_nlp.samplers.GreedySearch(end_token_id=END_ID)
    # Print the generated sequence (token ids).
    print(sampler(token_probability_fn, prompt, max_length=10))
    ```
    """

    def __init__(
        self,
        end_token_id=None,
        pad_token_id=0,
        jit_compile=True,
    ):
        self.end_token_id = end_token_id
        self.pad_token_id = pad_token_id
        self.jit_compile = jit_compile

    def _validate_prompt(self, prompt):
        """Helper method to validate input prompt."""
        if not isinstance(prompt, (tf.Tensor, tf.RaggedTensor)):
            prompt = tf.convert_to_tensor(prompt)
        return prompt

    def _validate_token_probability_fn(
        self, token_probability_fn, prompt, mask
    ):
        """Helper method to validate `token_probability_fn` output."""
        test_pred = token_probability_fn(prompt, mask=mask)
        if len(test_pred.shape) != 3:
            raise ValueError(
                "Output of `token_probability_fn` is not a 3D tensor, "
                "please provide a function with the output shape "
                "[batch_size, sequence_length, vocab_size]."
            )

    def _pad_prompt(self, prompt, max_length, pad_token_id):
        """Pad prompt to `max_length`."""
        mask = tf.ones_like(prompt, dtype=tf.bool)
        mask = mask.to_tensor(shape=(None, max_length))
        prompt = prompt.to_tensor(
            shape=(None, max_length), default_value=pad_token_id
        )
        return prompt, mask

    def _mask_tokens_after_end_token(
        self, prompt, max_length, end_token_id, pad_token_id
    ):
        """Helper function to mask the tokens after the end token."""
        # Mask out tokens after `end_token_id` is encountered.
        # Find index of first end_token_id.
        end_indices = tf.math.argmax(prompt == end_token_id, -1)
        # Use max_length if no `end_token_id` is found.
        end_indices = tf.where(
            end_indices == 0,
            tf.cast(max_length, dtype=end_indices.dtype),
            end_indices,
        )
        # Build a mask including end_token and replace tokens after end_token
        # with `pad_token_id`.
        mask_indices = tf.sequence_mask(end_indices + 1, maxlen=max_length)
        return tf.where(mask_indices, prompt, pad_token_id)

    def __call__(self, token_probability_fn, prompt, max_length):
        prompt = self._validate_prompt(prompt)

        input_is_1d = prompt.shape.rank == 1
        if input_is_1d:
            prompt = prompt[tf.newaxis, :]
        if isinstance(prompt, tf.Tensor):
            prompt = tf.RaggedTensor.from_tensor(
                prompt, padding=self.pad_token_id
            )
        shortest_prompt_len = tf.reduce_min(prompt.row_lengths())
        # Pad prompt to be a dense Tensor of shape [batch_size, max_length].
        # This step is required for XLA compatibility because XLA requires a
        # static shape, which means we cannot concatenate generated token to
        # current prompt.
        prompt, mask = self._pad_prompt(prompt, max_length, self.pad_token_id)
        self._validate_token_probability_fn(token_probability_fn, prompt, mask)

        # Convert `sample` method to a `tf.function`, and turn on
        # `jit_compile` accordingly.
        sample = tf.function(self.sample, jit_compile=self.jit_compile)
        prompt = sample(
            token_probability_fn, prompt, mask, max_length - shortest_prompt_len
        )

        # Mask out tokens after `end_token_id`.
        if self.end_token_id is not None:
            prompt = self._mask_tokens_after_end_token(
                prompt, max_length, self.end_token_id, self.pad_token_id
            )

        return tf.squeeze(prompt) if input_is_1d else prompt

    def sample(self, token_probability_fn, prompt, mask, num_steps):
        """Sampling logic implementation.

        Args:
            {{sample_keyword_docstring}}

        Returns:
            A dense int Tensor, representing the generated text in token id
            space.
        """
        raise NotImplementedError


base_sampler_keyword_args = """
    end_token_id: int, defaults to None. The token marking the end of the
        sequence, once encountered the generation is finished for the exact
        sequence. If None, every sequence is generated up to `max_length`.
        If set, all tokens after encountering `end_token_id` will be
        replaced with `pad_token_id`.
    pad_token_id: int, defaults to 0. The padding token.
    jit_compile: bool, defaults to True. If True, XLA compilation will be used.
    """

call_keyword_docstring = """
    token_probability_fn: a function that generates the probability of
        the next token over the whole vocabulary for each input token.
    prompt: a list of integers or an integer Tensor, can be 1D or 2D. The
        initial tokens to append generated tokens.
    max_length: int. The max length of generated sequence."""

sample_keyword_docstring = """
    token_probability_fn: a function that generates the probability of
        the next token over the whole vocabulary for each input token.
    prompt: a dense int Tensor of shape [batch_size, max_length]. The
        placeholder for generated sequence.
    num_steps: int. The remaining number of tokens to generate."""

Sampler.__doc__ = Sampler.__doc__.replace(
    "{{base_sampler_keyword_args}}", base_sampler_keyword_args
)
Sampler.__doc__ = Sampler.__doc__.replace(
    "{{call_keyword_docstring}}", call_keyword_docstring
)
Sampler.sample.__doc__ = Sampler.sample.__doc__.replace(
    "{{sample_keyword_docstring}}", sample_keyword_docstring
)
