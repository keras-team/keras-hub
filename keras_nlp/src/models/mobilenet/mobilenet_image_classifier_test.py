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
import numpy as np
import pytest

from keras_nlp.src.models.mobilenet.mobilenet_backbone import MobileNetBackbone
from keras_nlp.src.models.mobilenet.mobilenet_image_classifier import (
    MobileNetImageClassifier,
)
from keras_nlp.src.tests.test_case import TestCase


class MobileNetImageClassifierTest(TestCase):
    def setUp(self):
        # Setup model.
        self.images = np.ones((2, 224, 224, 3), dtype="float32")
        self.labels = [0, 3]
        self.backbone = MobileNetBackbone(
            stackwise_expansion=[1, 4, 6],
            stackwise_filters=[4, 8, 16],
            stackwise_kernel_size=[3, 3, 5],
            stackwise_stride=[2, 2, 1],
            stackwise_se_ratio=[0.25, None, 0.25],
            stackwise_activation=["relu", "relu", "hard_swish"],
            include_rescaling=False,
            output_filter=1280,
            activation="hard_swish",
            inverted_res_block=True,
            input_filter=16,
            image_shape=(224, 224, 3),
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
            cls=MobileNetImageClassifier,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 2),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=MobileNetImageClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
        )
