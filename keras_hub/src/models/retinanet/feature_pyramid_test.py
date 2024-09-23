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


from absl.testing import parameterized
from keras import ops

from keras_hub.src.models.retinanet.feature_pyramid import FeaturePyramid
from keras_hub.src.tests.test_case import TestCase


class FeaturePyramidTest(TestCase):
    @parameterized.named_parameters(
        (
            "equal_resolutions",
            3,
            7,
            {"P3": (2, 16, 16, 3), "P4": (2, 8, 8, 3), "P5": (2, 4, 4, 3)},
        ),
        (
            "different_resolutions",
            2,
            6,
            {
                "P2": (2, 64, 128, 4),
                "P3": (2, 32, 64, 8),
                "P4": (2, 16, 32, 16),
                "P5": (2, 8, 16, 32),
            },
        ),
    )
    def test_layer_output_shapes(self, min_level, max_level, input_shapes):
        layer = FeaturePyramid(min_level=min_level, max_level=max_level)

        inputs = {
            level: ops.ones(input_shapes[level]) for level in input_shapes
        }
        if layer.data_format == "channels_first":
            inputs = {
                level: ops.transpose(inputs[level], (0, 3, 1, 2))
                for level in inputs
            }

        output = layer(inputs)

        for level in inputs:
            self.assertEqual(
                output[level].shape,
                (
                    (input_shapes[level][0],)
                    + (layer.num_filters,)
                    + input_shapes[level][1:3]
                    if layer.data_format == "channels_first"
                    else input_shapes[level][:-1] + (layer.num_filters,)
                ),
            )
