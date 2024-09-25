import numpy as np

from keras_hub.src.bounding_box import validate_format
from keras_hub.src.tests.test_case import TestCase


class ValidateTest(TestCase):
    def test_raises_nondict(self):
        with self.assertRaisesRegex(
            ValueError, "Expected `bounding_boxes` to be a dictionary, got "
        ):
            validate_format.validate_format(np.ones((4, 3, 6)))

    def test_mismatch_dimensions(self):
        with self.assertRaisesRegex(
            ValueError,
            "Expected `boxes` and `classes` to have matching dimensions",
        ):
            validate_format.validate_format(
                {"boxes": np.ones((4, 3, 6)), "classes": np.ones((4, 6))}
            )

    def test_bad_keys(self):
        with self.assertRaisesRegex(ValueError, "containing keys"):
            validate_format.validate_format(
                {
                    "box": [
                        1,
                        2,
                        3,
                    ],
                    "class": [1234],
                }
            )
