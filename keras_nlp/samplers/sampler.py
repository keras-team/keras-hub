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
from tensorflow import keras

from keras_nlp.utils.python_utils import format_docstring

base_sampler_args_docstring = """
    jit_compile: bool, defaults to True. If True, XLA compilation will be used.
    """

call_args_docstring = """
    prompt: a list of integers or an integer Tensor, can be 1D or 2D. The
        initial tokens to append generated tokens.
    token_probability_fn: a function that generates the probability of
        the next token over the whole vocabulary for each input token.
    max_length: int. The max length of generated sequence.
    mask: a tensor, defaults to None. The padding mask of the prompt.
    end_token_id: int, defaults to None. The token marking the end of the
        sequence, once encountered the generation is finished for the exact
        sequence. If None, every sequence is generated up to `max_length`.
        If set, all tokens after encountering `end_token_id` will be
        replaced with `pad_token_id`.
    from_logits: bool, defaults to True. Indicate if the `token_probability_fn`
        returns logits. If False, `token_probability_fn` returns probability
        distributions.
    """

sample_args_docstring = """
    prompt: a dense int Tensor of shape [batch_size, max_length]. The
        placeholder for generated sequence.
    token_probability_fn: a function that generates the probability of
        the next token over the whole vocabulary for each input token.
    mask: a dense bool Tensor of shape [batch_size, max_length]. The mask of
        prompt.
    num_steps: int. The remaining number of tokens to generate.
    from_logits: bool, defaults to True. Indicate if the `token_probability_fn`
        returns logits. If False, `token_probability_fn` returns probability
        distributions.
    """


@format_docstring(
    base_sampler_args=base_sampler_args_docstring, call_args=call_args_docstring
)
@keras.utils.register_keras_serializable(package="keras_nlp")
class Sampler:
    """Base sampler class.

    Args:
        {{base_sampler_args}}

    Call Args:
        {{call_args}}

    The inputs and outputs of Sampler class are both token ids.

    Examples:

    Basic usage:
    ```python
    BATCH_SIZE = 8
    VOCAB_SIZE = 10
    FEATURE_SIZE = 16
    START_ID = 1
    END_ID = 2

    # Create a dummy model to predict the next token. Note that the output is
    # random without training, here we jsut demo how `samplers` works.
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
    print(sampler(prompt, token_probability_fn, 10, end_token_id=END_ID))
    ```

    Use with string inputs:
    ```python
    vocab = ["[UNK]", "[PAD]", "[END]", "the", "quick", "brown", "fox"]
    tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
        vocabulary=vocab,
        lowercase=True,
    )
    FEATURE_SIZE = 16
    VOCAB_SIZE = len(vocab)
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

    prompt = tokenizer("the quick brown fox")
    sampler = keras_nlp.samplers.Greedy()
    generated = sampler(
        prompt,
        token_probability_fn,
        10,
        end_token_id=tokenizer.token_to_id("[END]")
    )
    print(tokenizer.detokenize(generated))
    ```
    """

    def __init__(
        self,
        jit_compile=True,
    ):
        self.jit_compile = jit_compile

    def _validate_prompt_and_mask(self, prompt, mask):
        """Helper method to validate input prompt."""
        if not isinstance(prompt, (list, tf.RaggedTensor, tf.Tensor)):
            raise ValueError(
                "`prompt` must be one of `list`, `tf.RaggedTensor` or "
                f"`tf.Tensor`, but received: prompt={type(prompt)}."
            )

        if isinstance(prompt, tf.RaggedTensor):
            if mask:
                raise ValueError(
                    "`mask` is only valid when `prompt` is a list or dense "
                    f"tensor, but received type(prompt)={type(prompt)}."
                )
            return prompt, mask

        if isinstance(prompt, list):
            prompt = tf.convert_to_tensor(prompt)
        if not mask:
            mask = tf.cast(tf.ones_like(prompt), dtype=tf.bool)
        prompt = tf.ragged.boolean_mask(prompt, mask)
        return prompt, mask

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

    def _pad_prompt(self, prompt, max_length):
        """Pad prompt to `max_length`."""
        mask = tf.ones_like(prompt, dtype=tf.bool)
        mask = mask.to_tensor(shape=(None, max_length))
        prompt = prompt.to_tensor(shape=(None, max_length))
        return prompt, mask

    def _mask_tokens_after_end_token(
        self,
        prompt,
        max_length,
        end_token_id,
    ):
        """Helper function to truncate the tokens after the end token."""
        # Mask out tokens after `end_token_id` is encountered.
        # Find index of first end_token_id.
        end_indices = tf.math.argmax(prompt == end_token_id, -1)
        # Use max_length if no `end_token_id` is found.
        end_indices = tf.where(
            end_indices == 0,
            tf.cast(max_length, dtype=end_indices.dtype),
            end_indices,
        )
        # Truncate out tokens after (including) the end token.
        mask_indices = tf.sequence_mask(end_indices, maxlen=max_length)
        return tf.ragged.boolean_mask(prompt, mask_indices)

    def __call__(
        self,
        prompt,
        token_probability_fn,
        max_length,
        mask=None,
        end_token_id=None,
        from_logits=True,
    ):
        prompt, mask = self._validate_prompt_and_mask(prompt, mask)

        input_is_1d = prompt.shape.rank == 1
        if input_is_1d:
            prompt = tf.RaggedTensor.from_tensor(prompt[tf.newaxis, :])

        shortest_prompt_len = tf.reduce_min(prompt.row_lengths())
        # Pad prompt to be a dense Tensor of shape [batch_size, max_length].
        # This step is required for XLA compatibility because XLA requires a
        # static shape, which means we cannot concatenate generated token to
        # current prompt.
        prompt, mask = self._pad_prompt(prompt, max_length)
        self._validate_token_probability_fn(token_probability_fn, prompt, mask)

        # Convert `sample` method to a `tf.function`, and turn on
        # `jit_compile` accordingly.
        sample = tf.function(self.sample, jit_compile=self.jit_compile)
        prompt = sample(
            prompt,
            token_probability_fn,
            mask,
            max_length - shortest_prompt_len,
            from_logits,
        )

        # Mask out tokens after `end_token_id`.
        if end_token_id is not None:
            prompt = self._mask_tokens_after_end_token(
                prompt,
                max_length,
                end_token_id,
            )

        return tf.squeeze(prompt, axis=0) if input_is_1d else prompt

    @format_docstring(sample_args=sample_args_docstring)
    def sample(
        self, prompt, token_probability_fn, mask, num_steps, from_logits=True
    ):
        """Sampling logic implementation.

        Args:
            {{sample_args}}

        Returns:
            A dense int Tensor, representing the generated text in token id
            space.
        """
        raise NotImplementedError

    def get_config(self):
        return {
            "jit_compile": self.jit_compile,
        }
