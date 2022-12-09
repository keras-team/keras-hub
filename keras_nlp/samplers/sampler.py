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

    This class must be implemented by child class for instantiation.

    Args:
        {{base_optimizer_keyword_args}}

    Call Args:

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
        """Helper function to validate input to text_generation utils."""
        if not isinstance(prompt, (tf.Tensor, tf.RaggedTensor)):
            prompt = tf.convert_to_tensor(prompt)
        return prompt

    def _validate_token_probability_fn(
        self, token_probability_fn, prompt, mask
    ):
        """Helper function to validate `token_probability_fn` output."""
        test_pred = token_probability_fn(prompt, mask=mask)
        if len(test_pred.shape) != 3:
            raise ValueError(
                "Output of `token_probability_fn` is not a 3D tensor, "
                "please provide a function with the output shape "
                "[batch_size, sequence_length, vocab_size]."
            )

    def _align_and_pad_prompt(self, prompt, max_length, pad_token_id):
        """Align prompt to the right side, and pad to `max_length`."""
        longest_prompt_len = tf.reduce_max(prompt.row_lengths())
        pad_length = longest_prompt_len - prompt.row_lengths()

        prompt = tf.keras.utils.pad_sequences(
            prompt.to_list(), maxlen=longest_prompt_len, value=pad_token_id
        )

        mask = tf.RaggedTensor.from_row_lengths(
            tf.zeros(shape=[tf.reduce_sum(pad_length)], dtype=tf.int32),
            pad_length,
        )
        mask = mask.to_tensor(shape=(None, longest_prompt_len), default_value=1)

        shape = prompt.shape
        extra_space = tf.math.maximum(0, max_length - shape[1])
        pad_shape = [shape[0], extra_space]

        mask = tf.concat((mask, tf.zeros(pad_shape, tf.int32)), axis=1)
        prompt = tf.concat(
            (prompt, tf.zeros(pad_shape, prompt.dtype) + pad_token_id), axis=1
        )
        mask = tf.cast(mask, dtype=tf.bool)
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
        valid_indices = tf.sequence_mask(end_indices + 1, maxlen=max_length)
        return tf.where(valid_indices, prompt, pad_token_id)

    def __call__(self, token_probability_fn, prompt, max_length):
        """Sampling method to be called by users."""

        prompt = self._validate_prompt(prompt)

        input_is_1d = prompt.shape.rank == 1
        if input_is_1d:
            prompt = prompt[tf.newaxis, :]
        if isinstance(prompt, tf.Tensor):
            prompt = tf.RaggedTensor.from_tensor(
                prompt, padding=self.pad_token_id
            )
        longest_prompt_len = tf.reduce_max(prompt.row_lengths())
        prompt, mask = self._align_and_pad_prompt(
            prompt, max_length, self.pad_token_id
        )
        self._validate_token_probability_fn(token_probability_fn, prompt, mask)

        sample = tf.function(self.sample, jit_compile=self.jit_compile)
        prompt = sample(
            token_probability_fn, prompt, mask, max_length - longest_prompt_len
        )

        if self.end_token_id is not None:
            prompt = self._mask_tokens_after_end_token(
                prompt, max_length, self.end_token_id, self.pad_token_id
            )

        return tf.squeeze(prompt) if input_is_1d else prompt

    def sample(self, token_probability_fn, prompt, mask, num_steps):
        """Sampler's logic implementation.

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
    pad_token_id: int, defaults to 0. The pad token after `end_token_id`
        is received.
    jit_compile: bool, defaults to True. If using XLA compilation."""

call_keyword_docstring = """
    token_probability_fn: a function that generates the probability of
        the next token over the whole vocabulary for each input token.
    prompt: a list of integers or an integer Tensor, can be 1D or 2D. The
        initial tokens to append generated tokens.
    max_length: int. The max length of generated sequence."""

sample_keyword_docstring = """
    token_probability_fn: a function that generates the probability of
        the next token over the whole vocabulary for each input token.
    prompt: a list of integers or an integer Tensor, can be 1D or 2D. The
        initial tokens to append generated tokens.
    num_steps: int. The number of tokens to generate."""

Sampler.__doc__ = Sampler.__doc__.replace(
    "{{base_sampler_keyword_args}}", base_sampler_keyword_args
)
Sampler.__doc__ = Sampler.__call__.__doc__.replace(
    "{{call_keyword_docstring}}", call_keyword_docstring
)
Sampler.sample.__doc__ = Sampler.sample.__doc__.replace(
    "{{sample_keyword_docstring}}", sample_keyword_docstring
)
