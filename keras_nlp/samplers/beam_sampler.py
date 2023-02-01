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

from keras_nlp.samplers.sampler import Sampler
from keras_nlp.samplers.sampler import base_sampler_args_docstring
from keras_nlp.samplers.sampler import call_args_docstring
from keras_nlp.utils.python_utils import format_docstring


@format_docstring(
    base_sampler_args=base_sampler_args_docstring, call_args=call_args_docstring
)
@keras.utils.register_keras_serializable(package="keras_nlp")
class BeamSampler(Sampler):
    """Beam Sampler class.

    This sampler implements beam search algorithm. At each time-step, beam
    search keeps the beams (sequences) of the top `num_beams` highest
    accumulated probabilities, and uses each one of the beams to predict
    candidate next tokens.

    Args:
        num_beams: int. The number of beams that should be kept at each
            time-step. `num_beams` should be strictly positive.
        {{base_sampler_args}}

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
        jit_compile=True,
        run_eagerly=False,
    ):
        self.num_beams = num_beams
        super().__init__(jit_compile=jit_compile, run_eagerly=run_eagerly)

    def get_next_token(self, next_token_probs):
        # Beam search overrides the whole `sample` method.
        pass

    def sample(
        self, prompt, token_probability_fn, mask, num_steps, from_logits=True
    ):
        """Sampling logic implementation.

        Because beam search uses a different loop body, we have to override the
        whole `sample` method instead of just the `get_next_token` method.
        """
        batch_size, max_length = tf.shape(prompt)[0], tf.shape(prompt)[1]
        max_length = tf.cast(max_length, num_steps.dtype)
        length = max_length - num_steps
        dummy_preds = token_probability_fn(prompt, mask=mask)
        vocab_size = tf.shape(dummy_preds)[-1]
        pred_dtype = dummy_preds.dtype

        num_beams = self.num_beams

        # Initialize beam with shape `(batch_size, num_beams, length)`.
        beams = tf.repeat(tf.expand_dims(prompt, axis=1), num_beams, axis=1)
        # Initialize `beams_prob` with shape `(batch_size, num_beams)`.
        beams_prob = tf.zeros([batch_size, 1], dtype=pred_dtype)
        beams_prob = tf.concat(
            [beams_prob, tf.fill((batch_size, num_beams - 1), pred_dtype.min)],
            axis=-1,
        )

        def one_step(beams, beams_prob, length, mask):
            flattened_beams = tf.reshape(
                beams, shape=[batch_size * num_beams, -1]
            )
            repeated_mask = tf.tile(mask, [num_beams, 1])
            probs = token_probability_fn(flattened_beams, repeated_mask)
            preds = tf.gather(
                probs,
                tf.repeat(length - 1, batch_size * num_beams),
                axis=1,
                batch_dims=1,
            )
            if from_logits:
                preds = keras.activations.softmax(preds, axis=-1)
            # Reshape `preds` to shape `(batch_size, num_beams * vocab_size)`.

            preds = tf.reshape(preds, shape=[batch_size, -1])

            cum_probs = tf.math.log(preds) + tf.repeat(
                beams_prob, repeats=vocab_size, axis=1
            )

            candidate_prob, candidate_indexes = tf.math.top_k(
                cum_probs, k=num_beams, sorted=False
            )

            candidate_beam_indexes = candidate_indexes // vocab_size
            next_token = candidate_indexes % vocab_size

            beams = tf.gather(
                beams, candidate_beam_indexes, axis=1, batch_dims=1
            )

            # Build a new column of updates to scatter into the beam tensor.
            next_token = tf.where(
                condition=mask[..., length, tf.newaxis],
                x=beams[..., length],
                y=next_token,
            )
            next_token = tf.reshape(next_token, shape=[-1])

            mask = tf.tensor_scatter_nd_update(
                tensor=mask,
                indices=tf.stack(
                    (
                        tf.cast(tf.range(batch_size), dtype=length.dtype),
                        tf.repeat(length, batch_size),
                    ),
                    axis=1,
                ),
                updates=tf.repeat(True, batch_size),
            )

            # Generate `(batch_index, beam_index)` tuples for each beam.
            beam_indices = tf.where(tf.ones((batch_size, num_beams), tf.bool))
            beam_indices = tf.cast(beam_indices, dtype=length.dtype)
            # Build a tensor of repeated `length` values.
            length_indices = tf.fill((batch_size * num_beams, 1), length)
            # Concatenate to a triplet of `(batch_index, beam_index, length)`.
            indices = tf.concat([beam_indices, length_indices], axis=-1)

            # Update `beams[:, :, length]` with `next_token`.
            beams = tf.tensor_scatter_nd_update(
                tensor=beams,
                indices=indices,
                updates=next_token,
            )

            beams_prob = candidate_prob

            length = tf.add(length, 1)
            return beams, beams_prob, length, mask

        # Run a while loop till text of length `max_length` has been generated.
        beams, beams_prob, length, mask = tf.while_loop(
            cond=lambda beams, beams_prob, length, mask: tf.less(
                length, max_length
            ),
            body=one_step,
            loop_vars=[beams, beams_prob, length, mask],
            # There is a strange issue that when `batch_size=1`, the first loop
            # iteration changes `beams_prob`'s shape from [1, None] to
            # [None, None], which does not happen for `batch_size>1`.
            # As a workaround, we set shape invariants.
            shape_invariants=[
                beams.get_shape(),
                tf.TensorShape([None, None]),
                length.get_shape(),
                mask.get_shape(),
            ],
        )

        # Get the beam with the maximum probability.
        max_indexes = tf.math.argmax(beams_prob, axis=-1)
        max_beams = tf.gather(
            beams, max_indexes[:, tf.newaxis], axis=1, batch_dims=1
        )

        return tf.squeeze(max_beams, axis=1)

    def get_config(self):
        config = super().get_config()

        config.update({"num_beams": self.num_beams})
        return config
