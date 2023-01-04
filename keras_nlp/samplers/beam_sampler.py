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
"""Greedy Sampler."""

import tensorflow as tf
from tensorflow import keras

from keras_nlp.samplers.sampler import Sampler
from keras_nlp.samplers.sampler import base_sampler_keyword_args
from keras_nlp.samplers.sampler import call_keyword_docstring
from keras_nlp.samplers.sampler import sample_keyword_docstring


class BeamSampler(Sampler):
    """Beam Sampler class.

    This sampler implements beam search algorithm.

    Args:
        {{base_sampler_keyword_args}}

    Call Args:
        {{call_keyword_args}}
    """

    def __init__(
        self,
        num_beams,
        seed=None,
        from_logits=False,
        end_token_id=None,
        pad_token_id=0,
        jit_compile=True,
    ):
        self.num_beams = num_beams
        self.seed = seed
        self.from_logits = from_logits
        super().__init__(end_token_id, pad_token_id, jit_compile)

    def sample(self, token_probability_fn, prompt, mask, num_steps):
        """Sampler's logic implementation.

        Args:
            {{call_keyword_docstring}}
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
            if self.from_logits:
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
            loop_vars=(beams, beams_prob, length, mask),
        )

        # Get the beam with the maximum probability.
        max_indexes = tf.math.argmax(beams_prob, axis=-1)
        max_beams = tf.gather(
            beams, max_indexes[:, tf.newaxis], axis=1, batch_dims=1
        )
        prompt = tf.squeeze(max_beams)

        return prompt


BeamSampler.__doc__ = BeamSampler.__doc__.replace(
    "{{base_sampler_keyword_args}}", base_sampler_keyword_args
)
BeamSampler.__doc__ = BeamSampler.__doc__.replace(
    "{{call_keyword_docstring}}", call_keyword_docstring
)
BeamSampler.sample.__doc__ = BeamSampler.sample.__doc__.replace(
    "{{sample_keyword_docstring}}", sample_keyword_docstring
)
