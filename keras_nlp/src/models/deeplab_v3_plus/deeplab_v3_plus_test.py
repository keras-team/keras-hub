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

import numpy as np
import pytest

from keras_nlp.src.models.deeplab_v3_plus.deeplab_v3_plus_segmenter import (
    DeepLabV3Plus,
)
from keras_nlp.src.models.resnet.resnet_backbone import ResNetBackbone
from keras_nlp.src.tests.test_case import TestCase


class DeepLabV3PlusTest(TestCase):
    def setUp(self):
        self.init_kwargs = {
            "backbone": ResNetBackbone.from_preset(
                "hf://timm/resnet18.a1_in1k"
            ),
            "num_classes": 2,
            "low_level_feature_key": "P2",
            "spatial_pyramid_pooling_key": "P5",
            "projection_filters": 48,
            "spatial_pyramid_pooling": None,
            "dilation_rates": [6, 12, 18],
            "segmentation_head": None,
        }
        self.images = np.ones((2, 96, 96, 3), dtype="float32")
        self.labels = np.zeros((2, 96, 96, 2), dtype="float32")

    def test_segmentation_basics(self):
        self.run_segmentation_test(
            cls=DeepLabV3Plus,
            init_kwargs=self.init_kwargs,
            train_data=(self.images, self.labels),
            expected_output_shape=(2, 96, 96, 2),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=DeepLabV3Plus,
            init_kwargs=self.init_kwargs,
        )
