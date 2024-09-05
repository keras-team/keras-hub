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
from keras import ops

from keras_nlp.src.layers.preprocessing.resizing_image_converter import (
    ResizingImageConverter,
)
from keras_nlp.src.tests.test_case import TestCase


class ResizingImageConverterTest(TestCase):
    def test_resize_one(self):
        converter = ResizingImageConverter(22, 22)
        test_image = np.random.rand(10, 10, 3) * 255
        shape = ops.shape(converter(test_image))
        self.assertEqual(shape, (22, 22, 3))

    def test_resize_batch(self):
        converter = ResizingImageConverter(12, 12)
        test_batch = np.random.rand(4, 10, 20, 3) * 255
        shape = ops.shape(converter(test_batch))
        self.assertEqual(shape, (4, 12, 12, 3))

    def test_config(self):
        converter = ResizingImageConverter(
            width=12,
            height=20,
            pad_to_aspect_ratio=True,
            crop_to_aspect_ratio=False,
            fill_value=7.0,
        )
        clone = ResizingImageConverter.from_config(converter.get_config())
        test_batch = np.random.rand(4, 10, 20, 3) * 255
        self.assertAllClose(converter(test_batch), clone(test_batch))
