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
"""Base sampler class."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.compiler.tf2xla.python.xla import dynamic_update_slice

from keras_nlp.utils.python_utils import format_docstring

base_sampler_args_docstring = """
    jit_compile: bool, defaults to True. If True, XLA compilation will be used.
    run_eagerly: bool, defaults to False. If True, the sampler will run in
        the eager mode.
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
    cache: a dense int tensor, a cache of intermediate key and value tensor
        computed by each decoder self-attention layer. These values will only
        be computed once for each new token in the generated sequence.
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

    Subclassers should always implement the `get_next_token()` method, which
    gets the next token based on probability distribution over vocab tokens.
    Please check available subclass samplers for examples. If you need more
    control over the sampling process, please implement `sample()` method
    instead, see `keras_nlp.samplers.BeamSampler` for examples.

    Examples:

    Basic usage:
    ```python
    VOCAB_SIZE = 10

    # Create a dummy model to predict the next token. Note that the output is
    # random without training, here we just demo how `samplers` works.
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
    print(sampler(prompt, token_probability_fn, max_length=10, end_token_id=2))
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
    sampler = keras_nlp.samplers.GreedySampler()
    generated = sampler(
        prompt,
        token_probability_fn,
        max_length=10,
        end_token_id=tokenizer.token_to_id("[END]")
    )
    print(tokenizer.detokenize(generated))
    ```
    """

    def __init__(
        self,
        jit_compile=True,
        run_eagerly=False,
    ):
        if run_eagerly and jit_compile:
            raise ValueError(
                "XLA cannot be turned on under eager mode, received "
                "`jit_compile=True` and `run_eagerly=True`. Please either set "
                "`jit_compile=False` or set `run_eagerly=False`."
            )
        self.jit_compile = jit_compile
        self.run_eagerly = run_eagerly

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

    def _pad_prompt(self, prompt, max_length):
        """Pad prompt to `max_length`."""
        mask = tf.ones_like(prompt, dtype=tf.bool)
        mask = mask.to_tensor(shape=(None, max_length))
        prompt = prompt.to_tensor(shape=(None, max_length))
        return prompt, mask

    def _mask_tokens_after_end_token(
        self,
        generated_result,
        original_padding_mask,
        max_length,
        end_token_id,
    ):
        """Helper function to truncate the tokens after the end token."""
        # Create a tensor with True for each end_token_id.
        end_tokens = generated_result == end_token_id
        # Remove all end_token_ids in the original input.
        end_tokens = end_tokens & (original_padding_mask == tf.constant(False))
        # Find index of first end_token_id.
        end_indices = tf.math.argmax(end_tokens, -1)
        # Use max_length if no `end_token_id` is found.
        end_indices = tf.where(
            end_indices == 0,
            tf.cast(max_length, dtype=end_indices.dtype),
            end_indices,
        )
        # Truncate out tokens after (including) the end token.
        mask_indices = tf.sequence_mask(end_indices, maxlen=max_length)
        return tf.ragged.boolean_mask(generated_result, mask_indices)

    def __call__(
        self,
        prompt,
        token_probability_fn,
        max_length,
        mask=None,
        end_token_id=None,
        from_logits=True,
        cache=None,
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
        original_padding_mask = tf.identity(mask)

        # Convert `sample` method to a `tf.function` if `self.run_eagerly=False`
        # , and turn on `jit_compile` accordingly.
        sample = self.sample
        if not self.run_eagerly:
            sample = tf.function(self.sample, jit_compile=self.jit_compile)
        prompt = sample(
            prompt,
            token_probability_fn,
            mask,
            max_length - shortest_prompt_len,
            cache=cache,
            from_logits=from_logits,
        )
        # Mask out tokens after `end_token_id`.
        if end_token_id is not None:
            prompt = self._mask_tokens_after_end_token(
                prompt,
                original_padding_mask,
                max_length,
                end_token_id,
            )

        return tf.squeeze(prompt, axis=0) if input_is_1d else prompt

    def get_next_token(self, next_token_probs):
        """Get the next token.

        Args:
            next_token_probs: a Tensor, the probability distribution for next
                token over all vocab tokens.

        Get the next token based on given probability distribution over tokens.
        Subclasses must implement this method.
        """
        raise NotImplementedError

    def sample(
        self,
        prompt,
        token_probability_fn,
        mask,
        num_steps,
        from_logits=True,
        cache=None,
    ):
        """Sampling logic implementation.

        Args:
            prompt: a dense int Tensor of shape [batch_size, max_length]. The
                placeholder for generated sequence.
            token_probability_fn: a function that generates the probability of
                the next token over the whole vocabulary for each input token.
            mask: a dense bool Tensor of shape [batch_size, max_length]. The
                mask of prompt.
            num_steps: int. The remaining number of tokens to generate.
            from_logits: bool, defaults to True. Indicate if the
                `token_probability_fn` returns logits. If False,
                `token_probability_fn` returns probability distributions.
            cache: a dense int tensor, the cache used in decoding. The cache
                stores the key and value of each
                `keras_nlp.layers.CachedMultiHeadAttention` layer to make the
                decoding faster by avoiding duplicated computation.

        Returns:
            A dense int Tensor, representing the generated text in token id
            space.
        """
        batch_size, max_length = tf.shape(prompt)[0], tf.shape(prompt)[1]
        num_steps = tf.cast(num_steps, tf.int32)
        max_length = tf.cast(max_length, tf.int32)
        # The index of the last non-padding token in prompt. Since all sequences
        # are aligned to the right side, the index is the same for all.
        current_index = max_length - num_steps

        def one_step(
            current_index,
            prompt,
            mask,
            cache=None,
        ):
            last_index = current_index - 1
            if cache is not None:
                probs, cache = token_probability_fn(
                    prompt,
                    mask,
                    cache=cache,
                    cache_index=last_index,
                )
                next_token_probs = tf.squeeze(probs, axis=1)
            else:
                probs = token_probability_fn(prompt, mask)
                next_token_probs = tf.gather(
                    probs,
                    tf.repeat(current_index - 1, batch_size),
                    axis=1,
                    batch_dims=1,
                )

            if from_logits:
                next_token_probs = keras.activations.softmax(
                    next_token_probs, axis=-1
                )
            next_token = self.get_next_token(next_token_probs)
            next_token = tf.cast(next_token, prompt.dtype)
            next_token = tf.where(
                mask[:, current_index],
                prompt[:, current_index],
                next_token,
            )
            next_token = next_token[:, tf.newaxis]
            next_mask = tf.fill([batch_size, 1], True)
            slice_start = [0, current_index]
            mask = dynamic_update_slice(mask, next_mask, slice_start)
            prompt = dynamic_update_slice(prompt, next_token, slice_start)
            current_index = tf.add(current_index, 1)
            if cache is None:
                return current_index, prompt, mask
            return [current_index, prompt, mask, cache]

        if cache is None:
            _, prompt, _ = tf.while_loop(
                cond=lambda current_index, prompt, mask: tf.less(
                    current_index, max_length
                ),
                body=one_step,
                loop_vars=[current_index, prompt, mask],
            )
            return prompt
        # Run a while loop till `max_length` of tokens has been generated.
        _, prompt, _, _ = tf.while_loop(
            cond=lambda current_index, prompt, mask, cache: tf.less(
                current_index, max_length
            ),
            body=one_step,
            loop_vars=[current_index, prompt, mask, cache],
        )
        return prompt

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        return {
            "jit_compile": self.jit_compile,
        }
