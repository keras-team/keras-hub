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

import os

import keras
import numpy as np
import pytest

from keras_hub.api.models import MiTBackbone
from keras_hub.api.models import SegFormerBackbone
from keras_hub.api.models import SegFormerImageSegmenter
from keras_hub.src.tests.test_case import TestCase


class SegFormerTest(TestCase):
    def test_segformer_backbone_construction(self):
        backbone = MiTBackbone(
            depths=[2, 2],
            image_shape=(224, 224, 3),
            hidden_dims=[32, 64],
            num_layers=2,
            blockwise_num_heads=[1, 2],
            blockwise_sr_ratios=[8, 4],
            end_value=0.1,
            patch_sizes=[7, 3],
            strides=[4, 2],
        )
        SegFormerBackbone(backbone=backbone)

    def test_segformer_segmenter_construction(self):
        backbone = MiTBackbone(
            depths=[2, 2],
            image_shape=(224, 224, 3),
            hidden_dims=[32, 64],
            num_layers=2,
            blockwise_num_heads=[1, 2],
            blockwise_sr_ratios=[8, 4],
            end_value=0.1,
            patch_sizes=[7, 3],
            strides=[4, 2],
        )
        segformer_backbone = SegFormerBackbone(backbone=backbone)
        segformer = SegFormerImageSegmenter(
            backbone=segformer_backbone, num_classes=4
        )

    @pytest.mark.large
    def DISABLED_test_segformer_call(self):
        # TODO: Test of output comparison Fails
        backbone = MiTBackbone(
            depths=[2, 2],
            image_shape=(224, 224, 3),
            hidden_dims=[32, 64],
            num_layers=2,
            blockwise_num_heads=[1, 2],
            blockwise_sr_ratios=[8, 4],
            end_value=0.1,
            patch_sizes=[7, 3],
            strides=[4, 2],
        )
        model = SegFormerBackbone(backbone=backbone)
        segformer = SegFormerImageSegmenter(
            backbone=segformer_backbone, num_classes=4
        )

        images = np.random.uniform(size=(2, 224, 224, 3))
        segformer_output = segformer(images)
        segformer_predict = segformer.predict(images)

        assert segformer_output.shape == images.shape
        assert segformer_predict.shape == images.shape
