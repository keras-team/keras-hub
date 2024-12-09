import numpy as np
import pytest
from keras import ops
from keras import random

from keras_hub.src import bounding_box
from keras_hub.src.tests.test_case import TestCase


def is_tensorflow_ragged(value):
    if hasattr(value, "__class__"):
        return (
            value.__class__.__name__ == "RaggedTensor"
            and "tensorflow.python." in str(value.__class__.__module__)
        )
    return False


class MaskInvalidDetectionsTest(TestCase):
    def test_correctly_masks_based_on_max_dets(self):
        bounding_boxes = {
            "boxes": random.uniform((4, 100, 4)),
            "num_detections": ops.array([2, 3, 4, 2]),
            "classes": random.uniform((4, 100)),
        }

        result = bounding_box.mask_invalid_detections(bounding_boxes)

        negative_one_boxes = result["boxes"][:, 5:, :]
        self.assertAllClose(
            negative_one_boxes,
            -np.ones_like(ops.convert_to_numpy(negative_one_boxes)),
        )

        preserved_boxes = result["boxes"][:, :2, :]
        self.assertAllClose(preserved_boxes, bounding_boxes["boxes"][:, :2, :])

        boxes_from_image_3 = result["boxes"][2, :4, :]
        self.assertAllClose(
            boxes_from_image_3, bounding_boxes["boxes"][2, :4, :]
        )

    @pytest.mark.tf_keras_only
    def test_ragged_outputs(self):
        bounding_boxes = {
            "boxes": np.stack(
                [
                    np.random.uniform(size=(10, 4)),
                    np.random.uniform(size=(10, 4)),
                ]
            ),
            "num_detections": np.array([2, 3]),
            "classes": np.stack(
                [np.random.uniform(size=(10,)), np.random.uniform(size=(10,))]
            ),
        }

        result = bounding_box.mask_invalid_detections(
            bounding_boxes, output_ragged=True
        )
        self.assertTrue(is_tensorflow_ragged(result["boxes"]))
        self.assertEqual(result["boxes"][0].shape[0], 2)
        self.assertEqual(result["boxes"][1].shape[0], 3)

    @pytest.mark.tf_keras_only
    def test_correctly_masks_confidence(self):
        bounding_boxes = {
            "boxes": np.stack(
                [
                    np.random.uniform(size=(10, 4)),
                    np.random.uniform(size=(10, 4)),
                ]
            ),
            "confidence": np.random.uniform(size=(2, 10)),
            "num_detections": np.array([2, 3]),
            "classes": np.stack(
                [np.random.uniform(size=(10,)), np.random.uniform(size=(10,))]
            ),
        }

        result = bounding_box.mask_invalid_detections(
            bounding_boxes, output_ragged=True
        )
        self.assertTrue(is_tensorflow_ragged(result["boxes"]))
        self.assertEqual(result["boxes"][0].shape[0], 2)
        self.assertEqual(result["boxes"][1].shape[0], 3)
        self.assertEqual(result["confidence"][0].shape[0], 2)
        self.assertEqual(result["confidence"][1].shape[0], 3)
