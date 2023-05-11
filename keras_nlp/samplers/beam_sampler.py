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
        return_all_beams: bool. When set to `True`, the sampler will return all
            beams and their respective probabilities score.

    Call arguments:
        {{call_args}}

    Examples:
    Return only the beam with the highest accumulated probability.
    ```python
    # Use a simple alphabet of lowercase characters with ids in range [0, 25].
    int_lookup = {i: chr(i + ord('a')) for i in range(26)}
    char_lookup = {v: k for k, v in int_lookup.items()}
    batch_size, length, vocab_size = 1, 12, len(int_lookup)

    def next(prompt, cache, index):
        prompt_batch_size = tf.shape(prompt)[0]
        hidden_states = tf.ones((prompt_batch_size, 10))
        # A uniform distribution over our alphabet.
        logits = tf.ones((prompt_batch_size, vocab_size))
        return logits, hidden_states, cache

    output = keras_nlp.samplers.BeamSampler()(
        next=next,
        prompt=tf.fill((batch_size, length), char_lookup["z"]),
        index=5,
    )
    print(["".join([int_lookup[i] for i in s]) for s in output.numpy()])
    # >>> ['zzzzzeeeeeee']
    ```

    Return all beams and their probabilities.
    ```python
    # Use a simple alphabet of lowercase characters with ids in range [0, 25].
    int_lookup = {i: chr(i + ord('a')) for i in range(26)}
    char_lookup = {v: k for k, v in int_lookup.items()}
    batch_size, length, vocab_size = 1, 8, len(int_lookup)

    def next(prompt, cache, index):
        prompt_batch_size = tf.shape(prompt)[0]
        hidden_states = tf.ones((prompt_batch_size, 10))
        # A uniform distribution over our alphabet.
        logits = tf.ones((batch_size, vocab_size))
        return logits, hidden_states, cache

    beams, probs = keras_nlp.samplers.BeamSampler(return_all_beams=True)(
        next=next,
        prompt=tf.fill((batch_size, length,), char_lookup['z']),
        index=5,
    )

    print(beams.shape)
    # >>> (1, 5, 8)
    print(probs.shape)
    # >>> (1, 5)
    print(["".join([int_lookup[i] for i in s]) for s in beams[0].numpy()])
    # >>> ['zzzzzeee', 'zzzzzeed', 'zzzzzeec', 'zzzzzeea', 'zzzzzeeb']
    ```
    """

    def __init__(
        self,
        num_beams=5,
        return_all_beams=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_beams = num_beams
        self.return_all_beams = return_all_beams

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

        if mask is None:
            mask = tf.zeros_like(prompt, dtype=tf.bool)
        else:
            mask = tf.cast(mask, dtype=tf.bool)
        # `tf.while_loop` will not accept `None` as a value for `loop_vars`.
        cache = () if cache is None else cache
        # Add extra sequences for each beam.
        prompt, mask = create_beams(prompt), create_beams(mask)
        cache = tf.nest.map_structure(create_beams, cache)
        # Setup the initial beam log-likelihoods.
        # On the first loop, make sure only the original beam is considered.
        log_probs = tf.constant([[0.0] + [-1e9] * (self.num_beams - 1)])
        log_probs = flatten_beams(tf.repeat(log_probs, batch_size, axis=0))

        def cond(prompt, cache, index, log_probs):
            if end_token_id is None:
                return True
            # Stop if all sequences have produced a *new* end_token_id.
            end_tokens = (prompt == end_token_id) & (~mask)
            prompt_done = tf.reduce_any(end_tokens, axis=-1)
            return not tf.reduce_all(prompt_done)

        def body(prompt, cache, index, log_probs):
            # Compute the softmax distribution for the next token.
            logits, _, cache = next(prompt, cache, index)
            vocab_size = tf.shape(logits)[-1]
            probs = keras.activations.softmax(logits / self.temperature)

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
            # Ensure shape is `[None]`, otherwise it causes issues after
            # converting to TFLite.
            next_token = tf.ensure_shape(next_token, [None])
            # We need `ensure_shape` as `top_k` will change the static shape.
            next_log_probs = flatten_beams(next_log_probs)
            log_probs = tf.ensure_shape(next_log_probs, log_probs.shape)

            def gather_beams(x):
                x = unflatten_beams(x)
                x = tf.gather(x, beam_indices, axis=1, batch_dims=1)
                return flatten_beams(x)

            prompt = gather_beams(prompt)
            cache = tf.nest.map_structure(gather_beams, cache)

            # Update each beam with the next token.
            next_token = tf.cast(next_token, prompt.dtype)
            # Don't overwrite anywhere mask is True.
            next_token = tf.where(mask[:, index], prompt[:, index], next_token)
            # Update the prompt with the next token.
            next_token = next_token[:, tf.newaxis]
            prompt = dynamic_update_slice(prompt, next_token, [0, index])
            # Return the iteration of the loop state.
            return (prompt, cache, index + 1, log_probs)

        prompt, _, _, log_probs = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=(prompt, cache, index, log_probs),
            maximum_iterations=(max_length - index),
        )

        all_prompts = unflatten_beams(prompt)
        all_log_probs = unflatten_beams(log_probs)

        if self.return_all_beams:
            sorted_indices = tf.argsort(
                all_log_probs, axis=-1, direction="DESCENDING"
            )
            sorted_log_probs = tf.gather(
                all_log_probs, sorted_indices, axis=-1, batch_dims=1
            )
            sorted_prompts = tf.gather(
                all_prompts, sorted_indices, axis=1, batch_dims=1
            )
            return sorted_prompts, sorted_log_probs
        else:
            # Gather the top beam at each batch index.
            top_beams = tf.math.argmax(all_log_probs, axis=-1)[:, tf.newaxis]
            prompt = tf.gather(all_prompts, top_beams, axis=1, batch_dims=1)
            return tf.squeeze(prompt, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_beams": self.num_beams,
                "return_all_beams": self.return_all_beams,
            }
        )
        return config
