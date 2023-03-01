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

call_args_docstring = """
    prompt: A 2D integer tensor with shape `(batch_size, max_length)`. This
        tensor will be iteratively updated column by column with new sampled
        values.
    next: A function which takes in the `prompt, state, index` of the
        current generation loop, and outputs a tuple `probs, state` with the
        probability for the next token and state for the next iteration.
    state: Optional. A tensor or nested structure of tensors that will be
        updated by each call to `next`. This can be used to cache computations
        from early iterations of the generative loop.
    index: Optional. The first index to start sampling at.
    mask: Optional. A 2D integer tensor with the same shape as `prompt`.
        Locations which are `True` in the mask are never updated during
        sampling. Often this will mark all ids in `prompt` which were present in
        the original input.
    end_token_id: Optional. The token marking the end of the sequence. If
        specified, sampling will stop as soon as all sequences in the prompt
        produce a `end_token_id` in a location where `mask` is `False`.
    from_logits: Optional. Set to `True` if the `next` function return softmax
        probabilities instead of logits.
    """


@format_docstring(call_args=call_args_docstring)
@keras_nlp_export("keras_nlp.samplers.Sampler")
class Sampler:
    """Base sampler class.

    Call Args:
        {{call_args}}

    This base class can be extended to implement different auto-regressive
    sampling methods. Subclasses can either:
     - Override the `get_next_token()` method, which computes the next token
       based on a probability distribution over all possible vocab entries.
     - Override `__call__`, if the sampling method need additional state beyond
       the next tokens probability distribution to sample a sequence.
    Please check available subclass samplers for examples.

    Examples:

    ```python
    int_lookup = {i: chr(i + ord('a')) for i in range(26)}
    char_lookup = {v: k for k, v in int_lookup.items()}
    vocab_size = len(int_lookup)
    batch_size = 1

    def next(prompt, state, index):
        # return a uniform distribution over our alphabet.
        probs = tf.ones((batch_size, vocab_size))
        return probs, state

    output = keras_nlp.samplers.TopKSampler(k=2)(
        prompt=tf.zeros((batch_size, 12,)),
        next=next,
    )
    print("".join([int_lookup[i] for i in output.numpy()[0].tolist()]))
    # >>> "aaaaaaaaaaaa"
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

    def __call__(
        self,
        prompt,
        next,
        index=0,
        state=None,
        mask=None,
        end_token_id=None,
        from_logits=True,
    ):
        max_length = tf.shape(prompt)[-1]
        mask = tf.zeros_like(prompt, dtype=tf.bool) if mask is None else mask
        # `tf.while_loop` will not accept `None` as a value for `loop_vars`.
        state = () if state is None else state

        def cond(prompt, state, index):
            if end_token_id is None:
                return True
            # Stop if all sequences have produced a *new* end_token_id.
            end_tokens = (prompt == end_token_id) & (~mask)
            prompt_done = tf.reduce_any(end_tokens, axis=-1)
            return not tf.reduce_all(prompt_done)

        def body(prompt, state, index):
            # Compute the softmax distribution for the next token.
            probs, state = next(prompt, state, index)
            probs = keras.activations.softmax(probs) if from_logits else probs

            # Compute the next token.
            next_token = self.get_next_token(probs)
            # Don't overwrite anywhere mask is True.
            next_token = tf.cast(next_token, prompt.dtype)
            next_token = tf.where(mask[:, index], prompt[:, index], next_token)
            # Update the prompt with the next token.
            next_token = next_token[:, tf.newaxis]
            prompt = dynamic_update_slice(prompt, next_token, [0, index])
            # Return the next prompt, state and incremented index.
            return (prompt, state, index + 1)

        prompt, _, _ = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=(prompt, state, index),
            maximum_iterations=(max_length - index),
        )
        return prompt

    def get_next_token(self, probs):
        """Get the next token.

        Args:
            probs: a Tensor, the probability distribution for next
                token over all vocab tokens.

        Get the next token based on given probability distribution over tokens.
        Subclasses must implement this method.
        """
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        return {}
