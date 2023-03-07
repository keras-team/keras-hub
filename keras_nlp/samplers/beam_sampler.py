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
    VOCAB_SIZE = 10

    # Create a dummy model to predict the next token.
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

    sampler = keras_nlp.samplers.BeamSampler(num_beams=3)
    # Print the generated sequence (token ids).
    print(sampler(prompt, token_probability_fn, max_length=10))
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
        prompt,
        next,
        index=0,
        state=None,
        mask=None,
        end_token_id=None,
        from_logits=True,
    ):
        batch_size, max_length = tf.shape(prompt)[0], tf.shape(prompt)[1]

        def add_beams(x):
            return tf.repeat(x, self.num_beams, axis=0)

        def flatten(x):
            flat_shape = [batch_size * self.num_beams] + x.shape.as_list()[2:]
            return tf.reshape(x, shape=flat_shape)

        def unflatten(x):
            unflat_shape = [batch_size, self.num_beams] + x.shape.as_list()[1:]
            return tf.reshape(x, shape=unflat_shape)

        def cond(prompt, state, index, beam_probs):
            if end_token_id is None:
                return True
            # Stop if all sequences have produced a *new* end_token_id.
            end_tokens = (prompt == end_token_id) & (~mask)
            prompt_done = tf.reduce_any(end_tokens, axis=-1)
            return not tf.reduce_all(prompt_done)

        def body(prompt, state, index, beam_probs):
            # Compute the softmax distribution for the next token.
            probs, state = next(prompt, state, index)
            probs = keras.activations.softmax(probs) if from_logits else probs

            # Compute the running log-likelihood of each new candidate.
            vocab_size = tf.shape(probs)[-1]
            cum_probs = tf.math.log(probs) + beam_probs[..., tf.newaxis]
            # Reshape `preds` to shape `(batch_size, num_beams * vocab_size)`.
            cum_probs = tf.reshape(cum_probs, shape=[batch_size, -1])

            # Compute the top beam indices and next tokens.
            next_probs, indices = tf.math.top_k(
                cum_probs, k=self.num_beams, sorted=False
            )
            beam_indices = indices // vocab_size
            next_token = flatten(indices % vocab_size)
            # We need `ensure_shape` as `top_k` will change the static shape.
            beam_probs = tf.ensure_shape(flatten(next_probs), beam_probs.shape)

            # Gather the correct prompt and state beams.
            prompt = unflatten(prompt)
            state = tf.nest.map_structure(unflatten, state)
            prompt = tf.gather(prompt, beam_indices, axis=1, batch_dims=1)
            state = tf.gather(state, beam_indices, axis=1, batch_dims=1)
            prompt = flatten(prompt)
            state = tf.nest.map_structure(flatten, state)

            # Update each beam with the next token.
            next_token = tf.cast(next_token, prompt.dtype)
            # Don't overwrite anywhere mask is True.
            next_token = tf.where(mask[:, index], prompt[:, index], next_token)
            # Update the prompt with the next token.
            next_token = next_token[:, tf.newaxis]
            prompt = dynamic_update_slice(prompt, next_token, [0, index])
            # Return the iteration of the loop state.
            return (prompt, state, index + 1, beam_probs)

        mask = tf.zeros_like(prompt, dtype=tf.bool) if mask is None else mask
        # `tf.while_loop` will not accept `None` as a value for `loop_vars`.
        state = () if state is None else state
        # Add extra sequences for each beam.
        prompt, mask = add_beams(prompt), add_beams(mask)
        state = tf.nest.map_structure(add_beams, state)
        # Setup the initial beam log-likelihoods.
        # On the first loop, make sure only the original beam is considered.
        beam_probs = tf.constant([[0.0] + [-1e9] * (self.num_beams - 1)])
        beam_probs = flatten(tf.repeat(beam_probs, batch_size, axis=0))

        prompt, _, _, beam_probs = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=(prompt, state, index, beam_probs),
            maximum_iterations=(max_length - index),
        )

        # Gather the top beams for each batch index.
        prompt, beam_probs = unflatten(prompt), unflatten(beam_probs)
        top_beams = tf.math.argmax(beam_probs, axis=-1)[:, tf.newaxis]
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
