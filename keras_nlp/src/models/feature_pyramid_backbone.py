# Copyright 2024 The KerasNLP Authors
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
from keras_nlp.src.api_export import keras_nlp_export
from keras_nlp.src.models.backbone import Backbone


@keras_nlp_export("keras_nlp.models.FeaturePyramidBackbone")
class FeaturePyramidBackbone(Backbone):
    @property
    def pyramid_outputs(self):
        """Intermediate model outputs for feature extraction.

        Format is a dictionary with string as key and layer name as value.
        The key represents the level of the feature output. A typical feature
        pyramid has multiple levels corresponding to scales such as
        `["P2", "P3", "P4", "P5"]`. Scale `Pn` represents a feature map `2^n`
        times smaller in width and height than the input image.

        Example:

        ```python
        {
            'P2': 'v1_stack0_block3_out',
            'P3': 'v1_stack1_block4_out',
            'P4': 'v1_stack2_block6_out',
            'P5': 'v1_stack3_block3_out',
        }
        ```
        """
        return getattr(self, "_pyramid_outputs", {})

    @pyramid_outputs.setter
    def pyramid_outputs(self, value):
        if not isinstance(value, dict):
            raise TypeError(
                "`pyramid_outputs` must be a dictionary. "
                f"Received: value={value} of type {type(value)}"
            )
        for k, v in value.items():
            if not isinstance(k, str):
                raise TypeError(
                    "The key of `pyramid_outputs` must be a string. "
                    f"Received: key={k} of type {type(k)}"
                )
            self.get_layer(name=v)  # assert by calling `get_layer`
        self._pyramid_outputs = value
