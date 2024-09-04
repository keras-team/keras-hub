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

from keras_nlp.src.bounding_box import validate_format
from keras_nlp.src.tests.test_case import TestCase


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
