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

from keras_nlp.src.models.vgg.vgg_backbone import VGGBackbone
from keras_nlp.src.models.vgg.vgg_image_classifier import VGGImageClassifier
from keras_nlp.src.tests.test_case import TestCase


class VGGImageClassifierTest(TestCase):
    def setUp(self):
        # Setup model.
        images = np.ones((2, 224, 224, 3), dtype="float32")
        labels = [0, 3]
        self.backbone = VGGBackbone(
            stackwise_num_repeats=[2, 2, 3, 3, 3],
            input_shape=(224, 224, 3),
            include_rescaling=False,
            pooling="avg",
        )
        self.init_kwargs = {
            "backbone": self.backbone,
            "num_classes": 4,
        }
        self.train_data = (
            images,
            labels,
        )

    def test_classifier_basics(self):
        pytest.skip(reason="enable after preprocessor flow is figured out")
        self.run_task_test(
            cls=VGGImageClassifier,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 2),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=VGGImageClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
        )