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
import pytest
from keras import ops

from keras_nlp.src.models.video_swin.video_swin_backbone import (
    VideoSwinBackbone,
)
from keras_nlp.src.models.video_swin.video_swin_video_classifier import (
    VideoSwinVideoClassifier,
)
from keras_nlp.src.tests.test_case import TestCase


class VideoSwinVideoClassifierTest(TestCase):
    def setUp(self):
        self.images = ops.ones((2, 8, 256, 256, 3))
        self.labels = [0, 3]
        self.backbone = VideoSwinBackbone(
            image_shape=(8, 256, 256, 3),
            include_rescaling=True,
        )
        self.init_kwargs = {
            "backbone": self.backbone,
            "num_classes": 2,
            "activation": "softmax",
        }
        self.train_data = (self.images, self.labels)

    def test_classifier_basics(self):
        pytest.skip(
            reason="TODO: enable after preprocessor flow is figured out"
        )
        self.run_task_test(
            cls=VideoSwinVideoClassifier,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 2),
        )

    def test_head_dtype(self):
        model = VideoSwinVideoClassifier(
            **self.init_kwargs, head_dtype="bfloat16"
        )
        self.assertEqual(model.output_dense.compute_dtype, "bfloat16")

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=VideoSwinVideoClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
        )
