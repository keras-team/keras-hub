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
import numpy as np
import pytest

from keras_hub.src.models.mix_transformer.mix_transformer_backbone import (
    MiTBackbone,
)
from keras_hub.src.models.mix_transformer.mix_transformer_classifier import (
    MiTImageClassifier,
)
from keras_hub.src.tests.test_case import TestCase


class MiTImageClassifierTest(TestCase):
    def setUp(self):
        # Setup model.
        self.images = np.ones((2, 16, 16, 3), dtype="float32")
        self.labels = [0, 3]
        self.backbone = MiTBackbone(
            depths=[2, 2, 2, 2],
            image_shape=(16, 16, 3),
            hidden_dims=[4, 8],
            num_layers=2,
            blockwise_num_heads=[1, 2],
            blockwise_sr_ratios=[8, 4],
            end_value=0.1,
            patch_sizes=[7, 3],
            strides=[4, 2],
        )
        self.init_kwargs = {
            "backbone": self.backbone,
            "num_classes": 2,
            "activation": "softmax",
        }
        self.train_data = (
            self.images,
            self.labels,
        )

    def test_classifier_basics(self):
        pytest.skip(
            reason="TODO: enable after preprocessor flow is figured out"
        )
        self.run_task_test(
            cls=MiTImageClassifier,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 2),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=MiTImageClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
        )
