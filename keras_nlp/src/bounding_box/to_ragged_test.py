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
from keras import backend

from keras_nlp.src.bounding_box import to_dense
from keras_nlp.src.bounding_box import to_ragged
from keras_nlp.src.tests.test_case import TestCase


class ToRaggedTest(TestCase):
    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="Only applies to backends which support raggeds",
    )
    def test_converts_to_ragged(self):
        bounding_boxes = {
            "boxes": np.array(
                [[[0, 0, 0, 0], [0, 0, 0, 0]], [[2, 3, 4, 5], [0, 1, 2, 3]]]
            ),
            "classes": np.array([[-1, -1], [-1, 1]]),
            "confidence": np.array([[0.5, 0.7], [0.23, 0.12]]),
        }
        bounding_boxes = to_ragged.to_ragged(bounding_boxes)

        self.assertEqual(bounding_boxes["boxes"][1].shape, [1, 4])
        self.assertEqual(bounding_boxes["classes"][1].shape, [1])
        self.assertEqual(
            bounding_boxes["confidence"][1].shape,
            [
                1,
            ],
        )

        self.assertEqual(bounding_boxes["classes"][0].shape, [0])
        self.assertEqual(bounding_boxes["boxes"][0].shape, [0, 4])
        self.assertEqual(
            bounding_boxes["confidence"][0].shape,
            [
                0,
            ],
        )

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="Only applies to backends which support raggeds",
    )
    def test_round_trip(self):
        original = {
            "boxes": np.array(
                [
                    [[0, 0, 0, 0], [-1, -1, -1, -1]],
                    [[-1, -1, -1, -1], [-1, -1, -1, -1]],
                ]
            ),
            "classes": np.array([[1, -1], [-1, -1]]),
            "confidence": np.array([[0.5, -1], [-1, -1]]),
        }
        bounding_boxes = to_ragged.to_ragged(original)
        bounding_boxes = to_dense.to_dense(bounding_boxes, max_boxes=2)

        self.assertEqual(bounding_boxes["boxes"][1].shape, [2, 4])
        self.assertEqual(bounding_boxes["classes"][1].shape, [2])
        self.assertEqual(bounding_boxes["classes"][0].shape, [2])
        self.assertEqual(bounding_boxes["boxes"][0].shape, [2, 4])
        self.assertEqual(bounding_boxes["confidence"][0].shape, [2])

        self.assertAllEqual(bounding_boxes["boxes"], original["boxes"])
        self.assertAllEqual(bounding_boxes["classes"], original["classes"])
        self.assertAllEqual(
            bounding_boxes["confidence"], original["confidence"]
        )

    @pytest.mark.skipif(
        backend.backend() == "tensorflow",
        reason="Only applies to backends which don't support raggeds",
    )
    def test_backend_without_raggeds_throws(self):
        bounding_boxes = {
            "boxes": np.array(
                [[[0, 0, 0, 0], [0, 0, 0, 0]], [[2, 3, 4, 5], [0, 1, 2, 3]]]
            ),
            "classes": np.array([[-1, -1], [-1, 1]]),
            "confidence": np.array([[0.5, 0.7], [0.23, 0.12]]),
        }

        with self.assertRaisesRegex(NotImplementedError, "support ragged"):
            to_ragged.to_ragged(bounding_boxes)
