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

import numpy as np
import pytest
from keras import ops

from keras_hub.api.models import MiTBackbone
from keras_hub.api.models import SegFormerBackbone
from keras_hub.src.tests.test_case import TestCase


class SegFormerTest(TestCase):
    def setUp(self):
        image_encoder = MiTBackbone(
            depths=[2, 2],
            image_shape=(224, 224, 3),
            hidden_dims=[32, 64],
            num_layers=2,
            blockwise_num_heads=[1, 2],
            blockwise_sr_ratios=[8, 4],
            max_drop_path_rate=0.1,
            patch_sizes=[7, 3],
            strides=[4, 2],
        )
        projection_filters = 256
        self.input_size = 224
        self.input_data = ops.ones((2, self.input_size, self.input_size, 3))

        self.init_kwargs = {
            "projection_filters": projection_filters,
            "image_encoder": image_encoder,
        }

    def test_segformer_backbone_construction(self):

        SegFormerBackbone(
            image_encoder=self.init_kwargs["image_encoder"],
            projection_filters=self.init_kwargs["projection_filters"],
        )

    @pytest.mark.large
    def test_segformer_call(self):
        segformer_backbone = SegFormerBackbone(
            image_encoder=self.init_kwargs["image_encoder"],
            projection_filters=self.init_kwargs["projection_filters"],
        )

        images = np.random.uniform(size=(2, 224, 224, 3))
        segformer_output = segformer_backbone(images)
        segformer_predict = segformer_backbone.predict(images)

        assert segformer_output.shape == images.shape
        assert segformer_predict.shape == images.shape

    def test_backbone_basics(self):

        self.run_vision_backbone_test(
            cls=SegFormerBackbone,
            init_kwargs={**self.init_kwargs},
            input_data=self.input_data,
            expected_output_shape=(2, 56, 56, 256),
        )

    def test_task(self):
        self.run_task_test(
            cls=SegFormerBackbone,
            init_kwargs={**self.init_kwargs},
            train_data=self.input_data,
            expected_output_shape=(2, 224, 224),
        )

    @pytest.mark.large
    def test_saved_model(self):
        self.run_model_saving_test(
            cls=SegFormerBackbone,
            init_kwargs={**self.init_kwargs},
            input_data=self.input_data,
        )
