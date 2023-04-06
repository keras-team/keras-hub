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
"""Contrastive Sampler."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.compiler.tf2xla.python.xla import dynamic_update_slice

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.samplers.sampler import Sampler


@keras_nlp_export("keras_nlp.samplers.ContrastiveSampler")
class ContrastiveSampler(Sampler):
    def __init__(
        self,
        k=5,
        alpha=0.5,
        seed=None,
    ):
        super().__init__()
        self.k = k
        self.alpha = alpha
        self.seed = seed

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
            x = tf.repeat(x, self.k, axis=0)
            flat_shape = [batch_size * self.k] + x.shape.as_list()[1:]
            return tf.reshape(x, shape=flat_shape)

        def flatten_beams(x):
            """Combine the beam dim and batch dim."""
            flat_shape = [batch_size * self.k] + x.shape.as_list()[2:]
            return tf.reshape(x, shape=flat_shape)

        def unflatten_beams(x):
            """Separate the beam dim and batch dim."""
            unflat_shape = [batch_size, self.k] + x.shape.as_list()[1:]
            return tf.reshape(x, shape=unflat_shape)

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
            logits, state = next(prompt, state, index)
            probabilities = keras.activations.softmax(logits)

            prompt_beams, mask_beams = create_beams(prompt), create_beams(mask)
            state_beams = tf.nest.map_structure(create_beams, state)

            top_k_probabilities, top_k_indices = tf.math.top_k(
                probabilities, k=self.k, sorted=False
            )
            next_token_probabilities = flatten_beams(top_k_probabilities)
            next_token = flatten_beams(top_k_indices)
            next_token = tf.cast(next_token, prompt.dtype)

            next_token = tf.where(
                mask_beams[:, index], prompt_beams[:, index], next_token
            )
            # Update the prompt with the next token.
            next_token = next_token[:, tf.newaxis]
            prompt_beams = dynamic_update_slice(
                prompt_beams, next_token, [0, index]
            )

            _, next_state_beams = next(prompt_beams, state_beams, index + 1)
            hidden_states = next_state_beams["hidden_states"]
            last_token_state = hidden_states[:, index, :][:, tf.newaxis, :]
            previous_states = state_beams["hidden_states"][:, :index, :]
            similarity_scores = self.similarity(
                previous_states, last_token_state
            )
            max_similarity_scores = tf.reduce_max(similarity_scores, axis=1)

            accumulated_scores = (
                (1 - self.alpha) * next_token_probabilities
                - self.alpha * max_similarity_scores
            )
            unflat_score = unflatten_beams(accumulated_scores)
            unflat_prompt = unflatten_beams(prompt_beams)

            win_token_indices = tf.math.argmax(unflat_score, axis=1)
            prompt = tf.gather(
                unflat_prompt, win_token_indices, axis=1, batch_dims=1
            )
            return (prompt, state, index + 1)

        prompt, _, _ = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=(prompt, state, index),
            maximum_iterations=(max_length - index),
        )
        return prompt

    def similarity(self, h1, h2):
        return tf.squeeze(tf.matmul(h1, h2, transpose_b=True), axis=-1) / (
            tf.norm(h1, axis=-1) * tf.norm(h2, axis=-1)
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "k": self.k,
                "alpha": self.alpha,
                "seed": self.seed,
            }
        )
        return config
