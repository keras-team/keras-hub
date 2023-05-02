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

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.utils.python_utils import format_docstring

call_args_docstring = """next: A function which takes in the
            `prompt, cache, index` of the current generation loop, and outputs
            a tuple `(logits, cache, hidden_states)` with `logits` being the
            logits of next token, `cache` for next iteration, and
            `hidden_states` being the representation of the token.
        prompt: A 2D integer tensor with shape `(batch_size, max_length)`. This
            tensor will be iteratively updated column by column with new sampled
            values, starting at `index`.
        cache: Optional. A tensor or nested structure of tensors that will be
            updated by each call to `next`. This can be used to cache
            computations from early iterations of the generative loop.
        index: Optional. The first index of `prompt` to start sampling at.
            Usually this is set as the length of the shortest non-padding
            sequence in `prompt`.
        mask: Optional. A 2D integer tensor with the same shape as `prompt`.
            Locations which are `True` in the mask are never updated during
            sampling. Usually used to mark all locations in the dense prompt
            tensor which were present in a user input.
        end_token_id: Optional. The token marking the end of the sequence. If
            specified, sampling will stop as soon as all sequences in the prompt
            produce a `end_token_id` in a location where `mask` is `False`.
"""


@format_docstring(call_args=call_args_docstring)
@keras_nlp_export("keras_nlp.samplers.Sampler")
class Sampler:
    """Base sampler class.

    Args:
        temperature: float. optional. defaults to '1.0'. Used to control the
            randomness of the sampling. The higher the temperature, the
            more diverse the samples.

    Call arguments:
        {{call_args}}

    This base class can be extended to implement different auto-regressive
    sampling methods. Subclasses can either:

    - Override the `get_next_token()` method, which computes the next token
      based on a probability distribution over all possible vocab entries.
    - Override `__call__`, if the sampling method needs additional information
      beyond the next tokens probability distribution to sample a sequence.

    Please check available subclass samplers for examples.

    Examples:

    ```python
    # Use a simple alphabet of lowercase characters with ids in range [0, 25].
    int_lookup = {i: chr(i + ord('a')) for i in range(26)}
    char_lookup = {v: k for k, v in int_lookup.items()}
    batch_size, length, vocab_size = 1, 12, len(int_lookup)

    def next(prompt, cache, index):
        # return a uniform distribution over our alphabet.
        logits = tf.ones((batch_size, vocab_size))
        return logits, None, cache

    output = keras_nlp.samplers.GreedySampler()(
        next=next,
        prompt=tf.fill((batch_size, length,), char_lookup['z']),
        index=5,
    )
    print(["".join([int_lookup[i] for i in s]) for s in output.numpy()])
    # >>> ['zzzzzaaaaaaa']
    ```
    """

    def __init__(
        self,
        temperature=1.0,
    ):
        self.temperature = temperature

    def __call__(
        self,
        next,
        prompt,
        cache=None,
        index=0,
        mask=None,
        end_token_id=None,
        hidden_states=None,
    ):
        max_length = tf.shape(prompt)[-1]
        # Make sure `max_length` and `index` are the same dtype.
        index = tf.cast(index, max_length.dtype)
        if mask is None:
            mask = tf.zeros_like(prompt, dtype=tf.bool)
        else:
            mask = tf.cast(mask, dtype=tf.bool)
        # `tf.while_loop` will not accept `None` as a value for `loop_vars`.
        cache = () if cache is None else cache

        def cond(prompt, cache, index):
            if end_token_id is None:
                return True
            # Stop if all sequences have produced a *new* end_token_id.
            end_tokens = (prompt == end_token_id) & (~mask)
            prompt_done = tf.reduce_any(end_tokens, axis=-1)
            return not tf.reduce_all(prompt_done)

        def body(prompt, cache, index):
            # Compute the softmax distribution for the next token.
            logits, _, cache = next(prompt, cache, index)
            probabilities = keras.activations.softmax(logits / self.temperature)
            # Compute the next token.
            next_token = self.get_next_token(probabilities)
            # Don't overwrite anywhere mask is True.
            next_token = tf.cast(next_token, prompt.dtype)
            # Ensure shape is `[None]`, otherwise it causes issues after
            # converting to TFLite.
            next_token = tf.ensure_shape(next_token, [None])
            next_token = tf.where(mask[:, index], prompt[:, index], next_token)
            # Update the prompt with the next token.
            next_token = next_token[:, tf.newaxis]
            prompt = dynamic_update_slice(prompt, next_token, [0, index])
            # Return the next prompt, cache and incremented index.
            return (prompt, cache, index + 1)

        prompt, _, _ = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=(prompt, cache, index),
            maximum_iterations=(max_length - index),
        )
        return prompt

    def get_next_token(self, probabilities):
        """Get the next token.
        Args:
            probabilities: a Tensor, the probability distribution for next
                token over all vocab tokens.
        Get the next token based on given probability distribution over tokens.
        Subclasses must implement this method.
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        return {"temperature": self.temperature}
