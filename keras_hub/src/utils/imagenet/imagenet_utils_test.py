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

from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.imagenet.imagenet_utils import (
    decode_imagenet_predictions,
)


class ImageNetUtilsTest(TestCase):
    def test_decode_imagenet_predictions(self):
        preds = np.array(
            [
                [0.0] * 997 + [0.5, 0.3, 0.2],
                [0.0] * 997 + [0.2, 0.3, 0.5],
            ]
        )
        labels = decode_imagenet_predictions(preds, top=3)
        self.assertEqual(
            labels,
            [
                [("bolete", 0.5), ("ear", 0.3), ("toilet_tissue", 0.2)],
                [("toilet_tissue", 0.5), ("ear", 0.3), ("bolete", 0.2)],
            ],
        )
        labels = decode_imagenet_predictions(
            preds, top=3, include_synset_ids=True
        )
        self.assertEqual(
            labels,
            [
                [
                    ("n13054560", "bolete", 0.5),
                    ("n13133613", "ear", 0.3),
                    ("n15075141", "toilet_tissue", 0.2),
                ],
                [
                    ("n15075141", "toilet_tissue", 0.5),
                    ("n13133613", "ear", 0.3),
                    ("n13054560", "bolete", 0.2),
                ],
            ],
        )
