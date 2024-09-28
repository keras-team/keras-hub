import pytest
import tensorflow as tf
from keras import backend

from keras_hub.src.bounding_box import to_dense
from keras_hub.src.tests.test_case import TestCase


class ToDenseTest(TestCase):
    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="Only applies to backends which support raggeds",
    )
    def test_converts_to_dense(self):
        bounding_boxes = {
            "boxes": tf.ragged.constant(
                [[[0, 0, 1, 1]], [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]]
            ),
            "classes": tf.ragged.constant([[0], [1, 2, 3]]),
        }
        bounding_boxes = to_dense.to_dense(bounding_boxes)
        self.assertEqual(bounding_boxes["boxes"].shape, [2, 3, 4])
        self.assertEqual(bounding_boxes["classes"].shape, [2, 3])
