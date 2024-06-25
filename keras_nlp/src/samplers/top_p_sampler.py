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

from keras import ops
from keras import random

from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.samplers.sampler import Sampler


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
        k: int. If set, this argument defines a
            heuristic "top-k" cutoff applied before the "top-p" sampling. All
            logits not in the top `k` will be discarded, and the remaining
            logits will be sorted to find a cutoff point for `p`. Setting this
            arg can significantly speed sampling up by reducing the number
            of tokens to sort. Defaults to `None`.
        seed: int. The random seed. Defaults to `None`.

    Call arguments:
        {{call_args}}

    Examples:
    ```python
    causal_lm = keras_nlp.models.GPT2CausalLM.from_preset("gpt2_base_en")

    # Pass by name to compile.
    causal_lm.compile(sampler="top_p")
    causal_lm.generate(["Keras is a"])

    # Pass by object to compile.
    sampler = keras_nlp.samplers.TopPSampler(p=0.1, k=1_000)
    causal_lm.compile(sampler=sampler)
    causal_lm.generate(["Keras is a"])
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
        self.seed_generator = random.SeedGenerator(seed)

    def get_next_token(self, probabilities):
        cutoff = ops.shape(probabilities)[1]
        if self.k is not None:
            # If `k` is set, only sample from top `k` tokens.
            cutoff = self.k
        sorted_preds, sorted_indices = ops.top_k(
            probabilities, k=cutoff, sorted=True
        )
        # Calculate cumulative probability distribution.
        cumulative_probabilities = ops.cumsum(sorted_preds, axis=-1)
        # Create a mask for the tokens to keep.
        keep_mask = cumulative_probabilities <= self.p
        # Shift to include the last token that exceed p.
        shifted_keep_mask = ops.concatenate(
            [ops.ones_like(keep_mask[:, :1]), keep_mask[:, :-1]], axis=-1
        )
        # Filter out unmasked tokens and sample from filtered distribution.
        probabilities = ops.where(
            shifted_keep_mask,
            sorted_preds,
            ops.zeros(ops.shape(sorted_preds), dtype=sorted_preds.dtype),
        )
        sorted_next_token = random.categorical(
            ops.log(probabilities),
            1,
            seed=self.seed_generator,
            dtype="int32",
        )
        output = ops.take_along_axis(sorted_indices, sorted_next_token, axis=-1)
        return ops.squeeze(output, axis=-1)

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
