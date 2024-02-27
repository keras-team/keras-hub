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
from keras_nlp.models.generative_task import GenerativeTask


class FalconCausalLM(GenerativeTask):
    def __init__(self, backbone, preprocessor=None, **kwargs):
        inputs = backbone.input
        hidden_states = backbone(input)
        outputs = backbone.token_embedding(hidden_states, reverse=True)

        # Instantiate using Functional API Model constructor.
        super().__init__(
            inputs=inputs,
            outputs=outputs,
            include_preprocessing=preprocessor is not None,
            **kwargs,
        )
        self.backbone = backbone
        self.preprocessor = preprocessor
        self.generate_function = None
        self._sampler = None

        # TODO: Add lm_head.
        # sequence_output = keras.layers.Dense(
        #     vocabulary_size,
        #     use_bias=False,
        #     name="lm_head",
        # )(outputs) # TODO: should I pass outputs to the lm_head?
