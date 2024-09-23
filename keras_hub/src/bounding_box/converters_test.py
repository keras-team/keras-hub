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

import itertools

import numpy as np
import pytest
import tensorflow as tf
from absl.testing import parameterized
from keras import backend

from keras_hub.src.bounding_box import converters
from keras_hub.src.bounding_box import to_dense
from keras_hub.src.bounding_box import to_ragged
from keras_hub.src.tests.test_case import TestCase


class ConvertersTestCase(TestCase):
    def setUp(self):
        xyxy_box = np.array(
            [[[10, 20, 110, 120], [20, 30, 120, 130]]], dtype="float32"
        )
        yxyx_box = np.array(
            [[[20, 10, 120, 110], [30, 20, 130, 120]]], dtype="float32"
        )
        rel_xyxy_box = np.array(
            [[[0.01, 0.02, 0.11, 0.12], [0.02, 0.03, 0.12, 0.13]]],
            dtype="float32",
        )
        rel_xyxy_box_ragged_images = np.array(
            [[[0.10, 0.20, 1.1, 1.20], [0.40, 0.6, 2.40, 2.6]]], dtype="float32"
        )
        rel_yxyx_box = np.array(
            [[[0.02, 0.01, 0.12, 0.11], [0.03, 0.02, 0.13, 0.12]]],
            dtype="float32",
        )
        rel_yxyx_box_ragged_images = np.array(
            [[[0.2, 0.1, 1.2, 1.1], [0.6, 0.4, 2.6, 2.4]]], dtype="float32"
        )
        center_xywh_box = np.array(
            [[[60, 70, 100, 100], [70, 80, 100, 100]]], dtype="float32"
        )
        xywh_box = np.array(
            [[[10, 20, 100, 100], [20, 30, 100, 100]]], dtype="float32"
        )
        rel_xywh_box = np.array(
            [[[0.01, 0.02, 0.1, 0.1], [0.02, 0.03, 0.1, 0.1]]], dtype="float32"
        )
        rel_xywh_box_ragged_images = np.array(
            [[[0.1, 0.2, 1, 1], [0.4, 0.6, 2, 2]]], dtype="float32"
        )

        self.ragged_images = tf.ragged.constant(
            [
                np.ones(shape=[100, 100, 3]),
                np.ones(shape=[50, 50, 3]),
            ],  # 2 images
            ragged_rank=2,
        )

        self.images = np.ones([2, 1000, 1000, 3])

        self.ragged_classes = tf.ragged.constant([[0], [0]], dtype="float32")

        self.boxes = {
            "xyxy": xyxy_box,
            "center_xywh": center_xywh_box,
            "rel_xywh": rel_xywh_box,
            "xywh": xywh_box,
            "rel_xyxy": rel_xyxy_box,
            "yxyx": yxyx_box,
            "rel_yxyx": rel_yxyx_box,
        }

        self.boxes_ragged_images = {
            "xyxy": xyxy_box,
            "center_xywh": center_xywh_box,
            "rel_xywh": rel_xywh_box_ragged_images,
            "xywh": xywh_box,
            "rel_xyxy": rel_xyxy_box_ragged_images,
            "yxyx": yxyx_box,
            "rel_yxyx": rel_yxyx_box_ragged_images,
        }

    @parameterized.named_parameters(
        *[
            (f"{source}_{target}", source, target)
            for (source, target) in itertools.permutations(
                [
                    "xyxy",
                    "center_xywh",
                    "rel_xywh",
                    "xywh",
                    "rel_xyxy",
                    "yxyx",
                    "rel_yxyx",
                ],
                2,
            )
        ]
        + [("xyxy_xyxy", "xyxy", "xyxy")]
    )
    def test_converters(self, source, target):
        source, target
        source_box = self.boxes[source]
        target_box = self.boxes[target]

        self.assertAllClose(
            converters.convert_format(
                source_box, source=source, target=target, images=self.images
            ),
            target_box,
        )

    @parameterized.named_parameters(
        *[
            (f"{source}_{target}", source, target)
            for (source, target) in itertools.permutations(
                [
                    "xyxy",
                    "center_xywh",
                    "rel_xywh",
                    "xywh",
                    "rel_xyxy",
                    "yxyx",
                    "rel_yxyx",
                ],
                2,
            )
        ]
        + [("xyxy_xyxy", "xyxy", "xyxy")]
    )
    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="Only applies to backends which support raggeds",
    )
    def test_converters_ragged_images(self, source, target):
        source_box = _raggify(self.boxes_ragged_images[source])
        target_box = _raggify(self.boxes_ragged_images[target])
        self.assertAllClose(
            converters.convert_format(
                source_box,
                source=source,
                target=target,
                images=self.ragged_images,
            ),
            target_box,
        )

    @parameterized.named_parameters(
        *[
            (f"{source}_{target}", source, target)
            for (source, target) in itertools.permutations(
                [
                    "xyxy",
                    "center_xywh",
                    "rel_xywh",
                    "xywh",
                    "rel_xyxy",
                    "yxyx",
                    "rel_yxyx",
                ],
                2,
            )
        ]
        + [("xyxy_xyxy", "xyxy", "xyxy")]
    )
    def test_converters_unbatched(self, source, target):
        source_box = self.boxes[source][0]
        target_box = self.boxes[target][0]

        self.assertAllClose(
            converters.convert_format(
                source_box, source=source, target=target, images=self.images[0]
            ),
            target_box,
        )

    def test_raises_with_different_image_rank(self):
        source_box = self.boxes["xyxy"][0]
        with self.assertRaises(ValueError):
            converters.convert_format(
                source_box, source="xyxy", target="xywh", images=self.images
            )

    def test_without_images(self):
        source_box = self.boxes["xyxy"]
        target_box = self.boxes["xywh"]
        self.assertAllClose(
            converters.convert_format(source_box, source="xyxy", target="xywh"),
            target_box,
        )

    def test_rel_to_rel_without_images(self):
        source_box = self.boxes["rel_xyxy"]
        target_box = self.boxes["rel_yxyx"]
        self.assertAllClose(
            converters.convert_format(
                source_box, source="rel_xyxy", target="rel_yxyx"
            ),
            target_box,
        )

    @parameterized.named_parameters(
        *[
            (f"{source}_{target}", source, target)
            for (source, target) in itertools.permutations(
                [
                    "xyxy",
                    "center_xywh",
                    "rel_xywh",
                    "xywh",
                    "rel_xyxy",
                    "yxyx",
                    "rel_yxyx",
                ],
                2,
            )
        ]
        + [("xyxy_xyxy", "xyxy", "xyxy")]
    )
    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="Only applies to backends which support raggeds",
    )
    def test_ragged_bounding_box(self, source, target):
        source_box = _raggify(self.boxes[source])
        target_box = _raggify(self.boxes[target])
        self.assertAllClose(
            converters.convert_format(
                source_box, source=source, target=target, images=self.images
            ),
            target_box,
        )

    @parameterized.named_parameters(
        *[
            (f"{source}_{target}", source, target)
            for (source, target) in itertools.permutations(
                [
                    "xyxy",
                    "center_xywh",
                    "rel_xywh",
                    "xywh",
                    "rel_xyxy",
                    "yxyx",
                    "rel_yxyx",
                ],
                2,
            )
        ]
        + [("xyxy_xyxy", "xyxy", "xyxy")]
    )
    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="Only applies to backends which support raggeds",
    )
    def test_ragged_bounding_box_ragged_images(self, source, target):
        source_box = _raggify(self.boxes_ragged_images[source])
        target_box = _raggify(self.boxes_ragged_images[target])
        self.assertAllClose(
            converters.convert_format(
                source_box,
                source=source,
                target=target,
                images=self.ragged_images,
            ),
            target_box,
        )

    @parameterized.named_parameters(
        *[
            (f"{source}_{target}", source, target)
            for (source, target) in itertools.permutations(
                [
                    "xyxy",
                    "center_xywh",
                    "rel_xywh",
                    "xywh",
                    "rel_xyxy",
                    "yxyx",
                    "rel_yxyx",
                ],
                2,
            )
        ]
        + [("xyxy_xyxy", "xyxy", "xyxy")]
    )
    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="Only applies to backends which support raggeds",
    )
    def test_ragged_bounding_box_with_image_shape(self, source, target):
        source_box = _raggify(self.boxes[source])
        target_box = _raggify(self.boxes[target])
        self.assertAllClose(
            converters.convert_format(
                source_box,
                source=source,
                target=target,
                image_shape=(1000, 1000, 3),
            ),
            target_box,
        )

    @parameterized.named_parameters(
        *[
            (f"{source}_{target}", source, target)
            for (source, target) in itertools.permutations(
                [
                    "xyxy",
                    "center_xywh",
                    "rel_xywh",
                    "xywh",
                    "rel_xyxy",
                    "yxyx",
                    "rel_yxyx",
                ],
                2,
            )
        ]
        + [("xyxy_xyxy", "xyxy", "xyxy")]
    )
    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="Only applies to backends which support raggeds",
    )
    def test_dense_bounding_box_with_ragged_images(self, source, target):
        source_box = _raggify(self.boxes_ragged_images[source])
        target_box = _raggify(self.boxes_ragged_images[target])
        source_bounding_boxes = {
            "boxes": source_box,
            "classes": self.ragged_classes,
        }
        source_bounding_boxes = to_dense.to_dense(source_bounding_boxes)

        result_bounding_boxes = converters.convert_format(
            source_bounding_boxes,
            source=source,
            target=target,
            images=self.ragged_images,
        )
        result_bounding_boxes = to_ragged.to_ragged(result_bounding_boxes)

        self.assertAllClose(
            result_bounding_boxes["boxes"],
            target_box,
        )


def _raggify(tensor):
    tensor = tf.squeeze(tensor, axis=0)
    tensor = tf.RaggedTensor.from_row_lengths(tensor, [1, 1])
    return tensor
