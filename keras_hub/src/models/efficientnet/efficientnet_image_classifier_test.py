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
import pytest
from keras import ops

from keras_hub.src.models.efficientnet.efficientnet_backbone import (
    EfficientNetBackbone,
)
from keras_hub.src.models.efficientnet.efficientnet_image_classifier import (
    EfficientNetImageClassifier,
)
from keras_hub.src.tests.test_case import TestCase


class EfficientNetImageClassifierTest(TestCase):
    def setUp(self):
        self.images = ops.ones((2, 16, 16, 3))
        self.labels = [0, 3]
        self.backbone = EfficientNetBackbone(
            input_conv_filters=[64],
            input_conv_kernel_sizes=[7],
            stackwise_num_filters=[64, 64, 64],
            stackwise_num_blocks=[2, 2, 2],
            stackwise_num_strides=[1, 2, 2],
            block_type="basic_block",
            use_pre_activation=True,
            image_shape=(16, 16, 3),
            include_rescaling=False,
        )
        self.init_kwargs = {
            "backbone": self.backbone,
            "num_classes": 2,
            "pooling": "avg",
            "activation": "softmax",
        }
        self.train_data = (self.images, self.labels)

    def test_classifier_basics(self):
        pytest.skip(
            reason="TODO: enable after preprocessor flow is figured out"
        )
        self.run_task_test(
            cls=EfficientNetImageClassifier,
            init_kwargs=self.init_kwargs,
            train_data=self.train_data,
            expected_output_shape=(2, 2),
        )

    def test_head_dtype(self):
        model = EfficientNetImageClassifier(
            **self.init_kwargs, head_dtype="bfloat16"
        )
        self.assertEqual(model.output_dense.compute_dtype, "bfloat16")

    @pytest.mark.large
    def test_smallest_preset(self):
        # Test that our forward pass is stable!
        image_batch = self.load_test_image()[None, ...]
        self.run_preset_test(
            cls=EfficientNetImageClassifier,
            preset="enet_b0_ra",
            input_data=image_batch,
            expected_output_shape=(1, 1000),
            expected_labels=[85],
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=EfficientNetImageClassifier,
            init_kwargs=self.init_kwargs,
            input_data=self.images,
        )

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in EfficientNetImageClassifier.presets:
            self.run_preset_test(
                cls=EfficientNetImageClassifier,
                preset=preset,
                init_kwargs={"num_classes": 2},
                input_data=self.images,
                expected_output_shape=(2, 2),
            )
