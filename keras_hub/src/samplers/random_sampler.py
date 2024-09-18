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


@keras_hub_export("keras_hub.samplers.RandomSampler")
class RandomSampler(Sampler):
    """Random Sampler class.

    This sampler implements random sampling. Briefly, random sampler randomly
    selects a token from the entire distribution of the tokens, with selection
    chance determined by the probability of each token.

    Args:
        seed: int. The random seed. Defaults to `None`.

    Call arguments:
        {{call_args}}

    Examples:
    ```python
    causal_lm = keras_hub.models.GPT2CausalLM.from_preset("gpt2_base_en")

    # Pass by name to compile.
    causal_lm.compile(sampler="random")
    causal_lm.generate(["Keras is a"])

    # Pass by object to compile.
    sampler = keras_hub.samplers.RandomSampler(temperature=0.7)
    causal_lm.compile(sampler=sampler)
    causal_lm.generate(["Keras is a"])
    ```
    """

    def __init__(
        self,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.seed = seed
        self.seed_generator = random.SeedGenerator(seed)

    def get_next_token(self, probabilities):
        # Sample the next token from the probability distribution.
        next_token_id = random.categorical(
            ops.log(probabilities),
            1,
            seed=self.seed_generator,
            dtype="int32",
        )
        return ops.squeeze(next_token_id, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "seed": self.seed,
            }
        )
        return config
