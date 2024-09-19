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

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.samplers.sampler import Sampler


@keras_hub_export("keras_hub.samplers.GreedySampler")
class GreedySampler(Sampler):
    """Greedy sampler class.

    This sampler is implemented on greedy search, i.e., always picking up the
    token of the largest probability as the next token.

    Examples:
    ```python
    causal_lm = keras_hub.models.GPT2CausalLM.from_preset("gpt2_base_en")

    # Pass by name to compile.
    causal_lm.compile(sampler="greedy")
    causal_lm.generate(["Keras is a"])

    # Pass by object to compile.
    sampler = keras_hub.samplers.GreedySampler()
    causal_lm.compile(sampler=sampler)
    causal_lm.generate(["Keras is a"])
    ```
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def get_next_token(self, probabilities):
        return ops.argmax(probabilities, axis=-1)
