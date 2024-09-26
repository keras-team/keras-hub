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
from keras import ops

from keras_hub.src.models.retinanet.retinanet_label_encoder import (
    RetinaNetLabelEncoder,
)
from keras_hub.src.tests.test_case import TestCase


class RetinaNetLabelEncoderTest(TestCase):
    def test_layer_behaviors(self):
        images_shape = (8, 128, 128, 3)
        boxes_shape = (8, 10, 4)
        classes_shape = (8, 10)
        self.run_layer_test(
            cls=RetinaNetLabelEncoder,
            init_kwargs={
                "bounding_box_format": "xyxy",
                "min_level": 3,
                "max_level": 7,
                "num_scales": 3,
                "aspect_ratios": [0.5, 1.0, 2.0],
                "anchor_size": 8,
            },
            input_data={
                "images": np.random.uniform(size=images_shape),
                "gt_boxes": np.random.uniform(
                    size=boxes_shape, low=0.0, high=1.0
                ),
                "gt_classes": np.random.uniform(
                    size=classes_shape, low=0, high=5
                ),
            },
            expected_output_shape=((8, 3069, 4), (8, 3069)),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=0,
            run_training_check=False,
            run_precision_checks=False,
        )

    def test_label_encoder_output_shapes(self):
        images_shape = (8, 128, 128, 3)
        boxes_shape = (8, 10, 4)
        classes_shape = (8, 10)

        images = np.random.uniform(size=images_shape)
        boxes = np.random.uniform(size=boxes_shape, low=0.0, high=1.0)
        classes = np.random.uniform(size=classes_shape, low=0, high=5)

        encoder = RetinaNetLabelEncoder(
            bounding_box_format="xyxy",
            min_level=3,
            max_level=7,
            num_scales=3,
            aspect_ratios=[0.5, 1.0, 2.0],
            anchor_size=8,
        )

        box_targets, class_targets = encoder(images, boxes, classes)

        self.assertEqual(box_targets.shape, (8, 3069, 4))
        self.assertEqual(class_targets.shape, (8, 3069))

    def test_all_negative_1(self):
        images_shape = (8, 128, 128, 3)
        boxes_shape = (8, 10, 4)
        classes_shape = (8, 10)

        images = np.random.uniform(size=images_shape)
        boxes = -np.ones(shape=boxes_shape, dtype="float32")
        classes = -np.ones(shape=classes_shape, dtype="float32")

        encoder = RetinaNetLabelEncoder(
            bounding_box_format="xyxy",
            min_level=3,
            max_level=7,
            num_scales=3,
            aspect_ratios=[0.5, 1.0, 2.0],
            anchor_size=8,
        )

        box_targets, class_targets = encoder(images, boxes, classes)

        self.assertFalse(ops.any(ops.isnan(box_targets)))
        self.assertFalse(ops.any(ops.isnan(class_targets)))
