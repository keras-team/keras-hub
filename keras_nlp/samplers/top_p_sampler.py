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
"""Top-p Sampler."""

import tensorflow as tf

from keras_nlp.api_export import keras_nlp_export
from keras_nlp.samplers.sampler import Sampler
from keras_nlp.samplers.sampler import call_args_docstring
from keras_nlp.utils.python_utils import format_docstring


@format_docstring(call_args=call_args_docstring)
@keras_nlp_export("keras_nlp.samplers.TopPSampler")
class TopPSampler(Sampler):
    """Top-P Sampler class.

    This sampler implements top-p search algorithm. Top-p search selects tokens
    from the smallest subset of output probabilities that sum to greater than
    `p`. Put in another way, top-p will first order token predictions by
    likelihood, and ignore all tokens after the cumulative probability of
    selected tokens exceeds `p`, then select a token from the remaining tokens.

    Args:
        p: float, the `p` value of top-p.
        k: int, defaults to None. If set, this argument defines a
            heuristic "top-k" cutoff applied before the "top-p" sampling. All
            logits not in the top `k` will be discarded, and the remaining
            logits will be sorted to find a cutoff point for `p`. Setting this
            arg can significantly speed sampling up by reducing the number
            of tokens to sort.
        seed: int, defaults to None. The random seed.

    Call arguments:
        {{call_args}}

    Examples:
    ```python
    # Use a simple alphabet of lowercase characters with ids in range [0, 25].
    int_lookup = {i: chr(i + ord('a')) for i in range(26)}
    char_lookup = {v: k for k, v in int_lookup.items()}
    batch_size, length, vocab_size = 1, 12, len(int_lookup)

    def next(prompt, cache, index):
        hidden_states = tf.ones((batch_size, 10))
        # A uniform distribution over our alphabet.
        logits = tf.ones((batch_size, vocab_size))
        return logits, hidden_states, cache

    output = keras_nlp.samplers.TopPSampler(p=0.1)(
        next=next,
        prompt=tf.fill((batch_size, length,), char_lookup['z']),
        index=5,
    )
    print(["".join([int_lookup[i] for i in s]) for s in output.numpy()])
    # >>> ['zzzzzbabcccb']
    ```
    """

    def __init__(
        self,
        p=0.1,
        k=None,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.p = p
        self.k = k
        self.seed = seed

    def get_next_token(self, probabilities):
        cutoff = tf.shape(probabilities)[1]
        if self.k is not None:
            # If `k` is set, only sample from top `k` tokens.
            cutoff = tf.math.minimum(cutoff, self.k)
        sorted_preds, sorted_indices = tf.math.top_k(
            probabilities, k=cutoff, sorted=True
        )
        # Calculate cumulative probability distribution.
        cumulative_probabilities = tf.math.cumsum(sorted_preds, axis=-1)
        # Create a mask for the tokens to keep.
        keep_mask = cumulative_probabilities <= self.p
        # Shift to include the last token that exceed p.
        shifted_keep_mask = tf.concat(
            [tf.ones_like(keep_mask[:, :1]), keep_mask[:, :-1]], axis=-1
        )
        # Filter out unmasked tokens and sample from filtered distribution.
        probabilities = tf.where(
            shifted_keep_mask,
            sorted_preds,
            tf.zeros(tf.shape(sorted_preds), dtype=sorted_preds.dtype),
        )
        sorted_next_token = tf.random.categorical(
            tf.math.log(probabilities), 1, seed=self.seed
        )
        return tf.gather_nd(sorted_indices, sorted_next_token, batch_dims=1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "p": self.p,
                "k": self.k,
                "seed": self.seed,
            }
        )
        return config
