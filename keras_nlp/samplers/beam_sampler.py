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
"""Beam Sampler."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.compiler.tf2xla.python.xla import dynamic_update_slice

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.samplers.sampler import Sampler
from keras_nlp.samplers.sampler import call_args_docstring
from keras_nlp.utils.python_utils import format_docstring


@format_docstring(call_args=call_args_docstring)
@keras_nlp_export("keras_nlp.samplers.BeamSampler")
class BeamSampler(Sampler):
    """Beam Sampler class.

    This sampler implements beam search algorithm. At each time-step, beam
    search keeps the beams (sequences) of the top `num_beams` highest
    accumulated probabilities, and uses each one of the beams to predict
    candidate next tokens.

    Args:
        num_beams: int. The number of beams that should be kept at each
            time-step. `num_beams` should be strictly positive.

    Call Args:
        {{call_args}}

    Examples:
    ```python
    # Use a simple alphabet of lowercase characters to [0, 26).
    int_lookup = {i: chr(i + ord('a')) for i in range(26)}
    char_lookup = {v: k for k, v in int_lookup.items()}
    batch_size, length, vocab_size = 1, 12, len(int_lookup)

    def next(prompt, state, index):
        # A uniform distribution over our alphabet.
        logits = tf.ones((batch_size, vocab_size))
        return logits, state

    output = keras_nlp.samplers.BeamSampler()(
        next=next,
        prompt=tf.fill((batch_size, length,), char_lookup['z']),
        index=5,
    )
    print(["".join([int_lookup[i] for i in s]) for s in output.numpy()])
    # >>> "zzzzzaaaaaaa"
    ```
    """

    def __init__(
        self,
        num_beams=5,
    ):
        super().__init__()
        self.num_beams = num_beams

    def __call__(
        self,
        next,
        prompt,
        state=None,
        index=0,
        mask=None,
        end_token_id=None,
    ):
        batch_size, max_length = tf.shape(prompt)[0], tf.shape(prompt)[1]
        # Make sure max length and start index are the same dtype.
        index = tf.cast(index, max_length.dtype)

        def create_beams(x):
            """Add initial beam state."""
            return tf.repeat(x, self.num_beams, axis=0)

        def flatten_beams(x):
            """Combine the beam dim and batch dim."""
            flat_shape = [batch_size * self.num_beams] + x.shape.as_list()[2:]
            return tf.reshape(x, shape=flat_shape)

        def unflatten_beams(x):
            """Separate the beam dim and batch dim."""
            unflat_shape = [batch_size, self.num_beams] + x.shape.as_list()[1:]
            return tf.reshape(x, shape=unflat_shape)

        mask = tf.zeros_like(prompt, dtype=tf.bool) if mask is None else mask
        # `tf.while_loop` will not accept `None` as a value for `loop_vars`.
        state = () if state is None else state
        # Add extra sequences for each beam.
        prompt, mask = create_beams(prompt), create_beams(mask)
        state = tf.nest.map_structure(create_beams, state)
        # Setup the initial beam log-likelihoods.
        # On the first loop, make sure only the original beam is considered.
        log_probs = tf.constant([[0.0] + [-1e9] * (self.num_beams - 1)])
        log_probs = flatten_beams(tf.repeat(log_probs, batch_size, axis=0))

        def cond(prompt, state, index, log_probs):
            if end_token_id is None:
                return True
            # Stop if all sequences have produced a *new* end_token_id.
            end_tokens = (prompt == end_token_id) & (~mask)
            prompt_done = tf.reduce_any(end_tokens, axis=-1)
            return not tf.reduce_all(prompt_done)

        def body(prompt, state, index, log_probs):
            # Compute the softmax distribution for the next token.
            logits, state = next(prompt, state, index)
            vocab_size = tf.shape(logits)[-1]
            probs = keras.activations.softmax(logits)

            # Compute the running log-likelihood of each new candidate.
            next_log_probs = tf.math.log(probs) + log_probs[..., tf.newaxis]
            # Reshape `preds` to shape `(batch_size, num_beams * vocab_size)`.
            next_log_probs = tf.reshape(next_log_probs, shape=[batch_size, -1])

            # Compute the top beam indices and next tokens.
            next_log_probs, indices = tf.math.top_k(
                next_log_probs, k=self.num_beams, sorted=False
            )
            beam_indices = indices // vocab_size
            next_token = flatten_beams(indices % vocab_size)
            # We need `ensure_shape` as `top_k` will change the static shape.
            next_log_probs = flatten_beams(next_log_probs)
            log_probs = tf.ensure_shape(next_log_probs, log_probs.shape)

            def gather_beams(x):
                x = unflatten_beams(x)
                x = tf.gather(x, beam_indices, axis=1, batch_dims=1)
                return flatten_beams(x)

            prompt = gather_beams(prompt)
            state = tf.nest.map_structure(gather_beams, state)

            # Update each beam with the next token.
            next_token = tf.cast(next_token, prompt.dtype)
            # Don't overwrite anywhere mask is True.
            next_token = tf.where(mask[:, index], prompt[:, index], next_token)
            # Update the prompt with the next token.
            next_token = next_token[:, tf.newaxis]
            prompt = dynamic_update_slice(prompt, next_token, [0, index])
            # Return the iteration of the loop state.
            return (prompt, state, index + 1, log_probs)

        prompt, _, _, log_probs = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=(prompt, state, index, log_probs),
            maximum_iterations=(max_length - index),
        )

        # Gather the top beam at each batch index.
        prompt, log_probs = unflatten_beams(prompt), unflatten_beams(log_probs)
        top_beams = tf.math.argmax(log_probs, axis=-1)[:, tf.newaxis]
        prompt = tf.gather(prompt, top_beams, axis=1, batch_dims=1)
        return tf.squeeze(prompt, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_beams": self.num_beams,
            }
        )
        return config
