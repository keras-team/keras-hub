import numpy as np
from keras import ops

from keras_hub.src.bounding_box import utils
from keras_hub.src.tests.test_case import TestCase


class BoundingBoxUtilTest(TestCase):
    def test_clip_to_image_standard(self):
        # Test xyxy format unbatched
        height = 256
        width = 256
        bounding_boxes = {
            "boxes": np.array([[200, 200, 400, 400], [100, 100, 300, 300]]),
            "classes": np.array([0, 0]),
        }
        image = ops.ones(shape=(height, width, 3))
        bounding_boxes = utils.clip_to_image(
            bounding_boxes, bounding_box_format="xyxy", images=image
        )
        boxes = bounding_boxes["boxes"]
        self.assertAllGreaterEqual(ops.convert_to_numpy(boxes), 0)
        (
            x1,
            y1,
            x2,
            y2,
        ) = ops.split(boxes, 4, axis=1)
        self.assertAllLessEqual(
            ops.convert_to_numpy(ops.concatenate([x1, x2], axis=1)), width
        )
        self.assertAllLessEqual(
            ops.convert_to_numpy(ops.concatenate([y1, y2], axis=1)), height
        )
        # Test relative format batched
        image = ops.ones(shape=(1, height, width, 3))

        bounding_boxes = {
            "boxes": np.array([[[0.2, -1, 1.2, 0.3], [0.4, 1.5, 0.2, 0.3]]]),
            "classes": np.array([[0, 0]]),
        }
        bounding_boxes = utils.clip_to_image(
            bounding_boxes, bounding_box_format="rel_xyxy", images=image
        )
        boxes = bounding_boxes["boxes"]
        self.assertAllLessEqual(ops.convert_to_numpy(boxes), 1)

    def test_clip_to_image_filters_fully_out_bounding_boxes(self):
        # Test xyxy format unbatched
        height = 256
        width = 256
        bounding_boxes = {
            "boxes": np.array([[257, 257, 400, 400], [100, 100, 300, 300]]),
            "classes": np.array([0, 0]),
        }
        image = ops.ones(shape=(height, width, 3))
        bounding_boxes = utils.clip_to_image(
            bounding_boxes, bounding_box_format="xyxy", images=image
        )

        self.assertAllEqual(
            bounding_boxes["boxes"],
            np.array([[-1, -1, -1, -1], [100, 100, 256, 256]]),
        ),
        self.assertAllEqual(
            bounding_boxes["classes"],
            np.array([-1, 0]),
        )

    def test_clip_to_image_filters_fully_out_bounding_boxes_negative_area(self):
        # Test xyxy format unbatched
        height = 256
        width = 256
        bounding_boxes = {
            "boxes": np.array([[110, 120, 100, 100], [100, 100, 300, 300]]),
            "classes": np.array([0, 0]),
        }
        image = ops.ones(shape=(height, width, 3))
        bounding_boxes = utils.clip_to_image(
            bounding_boxes, bounding_box_format="xyxy", images=image
        )
        self.assertAllEqual(
            bounding_boxes["boxes"],
            np.array(
                [
                    [
                        -1,
                        -1,
                        -1,
                        -1,
                    ],
                    [
                        100,
                        100,
                        256,
                        256,
                    ],
                ]
            ),
        )
        self.assertAllEqual(
            bounding_boxes["classes"],
            np.array([-1, 0]),
        )

    def test_clip_to_image_filters_nans(self):
        # Test xyxy format unbatched
        height = 256
        width = 256
        bounding_boxes = {
            "boxes": np.array(
                [[0, float("NaN"), 100, 100], [100, 100, 300, 300]]
            ),
            "classes": np.array([0, 0]),
        }
        image = ops.ones(shape=(height, width, 3))
        bounding_boxes = utils.clip_to_image(
            bounding_boxes, bounding_box_format="xyxy", images=image
        )
        self.assertAllEqual(
            bounding_boxes["boxes"],
            np.array(
                [
                    [
                        -1,
                        -1,
                        -1,
                        -1,
                    ],
                    [
                        100,
                        100,
                        256,
                        256,
                    ],
                ]
            ),
        )
        self.assertAllEqual(
            bounding_boxes["classes"],
            np.array([-1, 0]),
        )

    def test_is_relative_util(self):
        self.assertTrue(utils.is_relative("rel_xyxy"))
        self.assertFalse(utils.is_relative("xyxy"))

        with self.assertRaises(ValueError):
            _ = utils.is_relative("bad_format")

    def test_as_relative_util(self):
        self.assertEqual(utils.as_relative("yxyx"), "rel_yxyx")
        self.assertEqual(utils.as_relative("rel_xywh"), "rel_xywh")
