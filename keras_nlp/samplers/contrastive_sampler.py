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
from keras_nlp.samplers.sampler import call_args_docstring
from keras_nlp.utils.python_utils import format_docstring


@format_docstring(call_args=call_args_docstring)
@keras_nlp_export("keras_nlp.samplers.ContrastiveSampler")
class ContrastiveSampler(Sampler):
    """Contrastive Sampler class.

    This sampler implements contrastive search algorithm. In short, the sampler
    chooses the token having the max "score" as the next token. The "score" is
    a weighted sum between token's probability and max similarity against
    previous tokens. By using this joint score, contrastive sampler reduces the
    behavior of duplicating seen tokens.

    Args:
        k: int, the `k` value of top-k. Next token will be chosen from k tokens.
        alpha: float, the weight of minus max similarity in joint score
            computation. The larger the value of `alpha`, the score relies more
            on the similarity than the token probability.
        seed: int, defaults to None. The random seed.

    Call arguments:
        {{call_args}}

    Examples:
    ```python
    # Use a simple alphabet of lowercase characters to [0, 26).
    int_lookup = {i: chr(i + ord("a")) for i in range(26)}
    char_lookup = {v: k for k, v in int_lookup.items()}
    batch_size, length, vocab_size = 1, 12, len(int_lookup)
    hidden_size = 5
    index = 5

    def next(prompt, cache, index):
        prompt_batch_size = tf.shape(prompt)[0]
        hidden_states = tf.ones((prompt_batch_size, hidden_size))
        # A uniform distribution over our alphabet.
        logits = tf.ones((prompt_batch_size, vocab_size))
        return logits, hidden_states, cache

    output = keras_nlp.samplers.ContrastiveSampler()(
        next=next,
        prompt=tf.fill((batch_size, length), char_lookup["z"]),
        index=index,
        hidden_states=tf.ones([batch_size, index, hidden_size]),
    )
    print(["".join([int_lookup[i] for i in s]) for s in output.numpy()])
    # >>> "zzzzzeeeeeee"
    ```
    """

    def __init__(
        self,
        k=5,
        alpha=0.6,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.k = k
        self.alpha = alpha
        self.seed = seed

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
        if hidden_states is None:
            raise ValueError(
                "`ContrastiveSampler` requires passing a `hidden_states`, but"
                "received `None`."
            )
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
        # Compute initial logits.
        logits, _, cache = next(prompt, cache, index)
        # `tf.while_loop` will not accept `None` as a value for `loop_vars`.
        cache = () if cache is None else cache

        def cond(prompt, cache, index, logits, hidden_states):
            if end_token_id is None:
                return True
            # Stop if all sequences have produced a *new* end_token_id.
            end_tokens = (prompt == end_token_id) & (~mask)
            prompt_done = tf.reduce_any(end_tokens, axis=-1)
            return not tf.reduce_all(prompt_done)

        def body(prompt, cache, index, logits, hidden_states):
            # Compute the softmax distribution for the next token.
            probabilities = keras.activations.softmax(logits / self.temperature)

            # Replicate for `self.k` times to find the best token in top-k
            # candidates.
            prompt_beams = create_beams(prompt)
            mask_beams = create_beams(mask)
            hidden_states_beams = create_beams(hidden_states)
            cache_beams = tf.nest.map_structure(create_beams, cache)

            # Get top-k candidate tokens and their probabilities.
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

            # Compute the logits and hidden states for top-k candidate tokens.
            next_logits, next_hidden_states_beams, cache_beams = next(
                prompt_beams, cache_beams, index + 1
            )

            # Compute the max similarity score for top-k candidate tokens
            # against previous tokens.
            similarity_scores = self.similarity(
                hidden_states_beams, next_hidden_states_beams
            )
            max_similarity_scores = tf.cast(
                tf.reduce_max(similarity_scores[:, :index], axis=1),
                dtype=next_token_probabilities.dtype,
            )
            if index == 0:
                # If the index is 0, there is no previous states so we set
                # `max_similarity_scores` the same for all beams.
                max_similarity_scores = tf.zeros_like(max_similarity_scores)
            # The final score of each candidate token is weighted sum of
            # probability and similarity against previous tokens.
            accumulated_scores = (
                (1 - self.alpha) * next_token_probabilities
                - self.alpha * max_similarity_scores
            )
            # Unflatten varibles to shape [batch_size, self.k, ...] for
            # gather purpose.
            unflat_score = unflatten_beams(accumulated_scores)
            unflat_prompt = unflatten_beams(prompt_beams)
            unflat_next_logits = unflatten_beams(next_logits)
            unflat_next_hidden_states = unflatten_beams(
                next_hidden_states_beams
            )
            unflat_cache = tf.nest.map_structure(unflatten_beams, cache_beams)
            best_token_indices = tf.math.argmax(unflat_score, axis=1)

            def gather_best_token(beams):
                return tf.gather(
                    beams,
                    best_token_indices,
                    axis=1,
                    batch_dims=1,
                )

            prompt = gather_best_token(unflat_prompt)
            # We avoid recomputing forward pass for each token by updating the
            # cache/hidden_states using the output, and pass the logits to
            # next iteration step.
            logits = gather_best_token(unflat_next_logits)
            next_hidden_states = gather_best_token(unflat_next_hidden_states)
            cache = tf.nest.map_structure(gather_best_token, unflat_cache)

            hidden_states = dynamic_update_slice(
                hidden_states,
                next_hidden_states[:, tf.newaxis, :],
                [0, index, 0],
            )
            return (prompt, cache, index + 1, logits, hidden_states)

        prompt, _, _, _, _ = tf.while_loop(
            cond=cond,
            body=body,
            loop_vars=(prompt, cache, index, logits, hidden_states),
            maximum_iterations=(max_length - index),
        )
        return prompt

    def similarity(self, h1, h2):
        h2 = h2[..., tf.newaxis]
        return tf.squeeze(tf.matmul(h1, h2), axis=-1) / (
            tf.norm(h1, axis=-1) * tf.norm(h2, axis=-2)
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
