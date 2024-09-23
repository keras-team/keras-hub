# Copyright 2024 The KerasHub Authors
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

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.samplers.sampler import Sampler


@keras_hub_export("keras_hub.samplers.TopKSampler")
class TopKSampler(Sampler):
    """Top-K Sampler class.

    This sampler implements top-k search algorithm. Briefly, top-k algorithm
    randomly selects a token from the tokens of top K probability, with
    selection chance determined by the probability.

    Args:
        k: int, the `k` value of top-k.
        seed: int. The random seed. Defaults to `None`.

    Call arguments:
        {{call_args}}

    Examples:
    ```python
    causal_lm = keras_hub.models.GPT2CausalLM.from_preset("gpt2_base_en")

    # Pass by name to compile.
    causal_lm.compile(sampler="top_k")
    causal_lm.generate(["Keras is a"])

    # Pass by object to compile.
    sampler = keras_hub.samplers.TopKSampler(k=5, temperature=0.7)
    causal_lm.compile(sampler=sampler)
    causal_lm.generate(["Keras is a"])
    ```
    """

    def __init__(
        self,
        k=5,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.k = k
        self.seed = seed
        self.seed_generator = random.SeedGenerator(seed)

    def get_next_token(self, probabilities):
        # Filter out top-k tokens.
        top_k_pred, top_k_indices = ops.top_k(
            probabilities,
            k=self.k,
            sorted=False,
        )
        # Sample the next token from the probability distribution.
        sample_indices = random.categorical(
            # tf does not support half precision multinomial sampling, so make
            # sure we have full precision here.
            ops.cast(ops.log(top_k_pred), "float32"),
            1,
            seed=self.seed_generator,
            dtype="int32",
        )

        # Rearrange to get the next token idx from the original order.
        output = ops.take_along_axis(top_k_indices, sample_indices, axis=-1)
        return ops.squeeze(output, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "k": self.k,
                "seed": self.seed,
            }
        )
        return config
