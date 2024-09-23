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

from keras_hub.src.layers.preprocessing.resizing_image_converter import (
    ResizingImageConverter,
)
from keras_hub.src.tests.test_case import TestCase


class ResizingImageConverterTest(TestCase):
    def test_resize_simple(self):
        converter = ResizingImageConverter(height=4, width=4)
        inputs = np.ones((10, 10, 3))
        outputs = converter(inputs)
        self.assertAllClose(outputs, ops.ones((4, 4, 3)))

    def test_resize_one(self):
        converter = ResizingImageConverter(
            height=4,
            width=4,
            mean=(0.5, 0.7, 0.3),
            variance=(0.25, 0.1, 0.5),
            scale=1 / 255.0,
        )
        inputs = np.ones((10, 10, 3)) * 128
        outputs = converter(inputs)
        self.assertEqual(ops.shape(outputs), (4, 4, 3))
        self.assertAllClose(outputs[:, :, 0], np.ones((4, 4)) * 0.003922)
        self.assertAllClose(outputs[:, :, 1], np.ones((4, 4)) * -0.626255)
        self.assertAllClose(outputs[:, :, 2], np.ones((4, 4)) * 0.285616)

    def test_resize_batch(self):
        converter = ResizingImageConverter(
            height=4,
            width=4,
            mean=(0.5, 0.7, 0.3),
            variance=(0.25, 0.1, 0.5),
            scale=1 / 255.0,
        )
        inputs = np.ones((2, 10, 10, 3)) * 128
        outputs = converter(inputs)
        self.assertEqual(ops.shape(outputs), (2, 4, 4, 3))
        self.assertAllClose(outputs[:, :, :, 0], np.ones((2, 4, 4)) * 0.003922)
        self.assertAllClose(outputs[:, :, :, 1], np.ones((2, 4, 4)) * -0.626255)
        self.assertAllClose(outputs[:, :, :, 2], np.ones((2, 4, 4)) * 0.285616)

    def test_errors(self):
        with self.assertRaises(ValueError):
            ResizingImageConverter(
                height=4,
                width=4,
                mean=(0.5, 0.7, 0.3),
            )

    def test_config(self):
        converter = ResizingImageConverter(
            width=12,
            height=20,
            mean=(0.5, 0.7, 0.3),
            variance=(0.25, 0.1, 0.5),
            scale=1 / 255.0,
            crop_to_aspect_ratio=False,
            interpolation="nearest",
        )
        clone = ResizingImageConverter.from_config(converter.get_config())
        test_batch = np.random.rand(4, 10, 20, 3) * 255
        self.assertAllClose(converter(test_batch), clone(test_batch))
