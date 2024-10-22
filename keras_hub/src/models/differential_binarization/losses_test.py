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

from keras_hub.src.models.differential_binarization.losses import DBLoss
from keras_hub.src.models.differential_binarization.losses import DiceLoss
from keras_hub.src.models.differential_binarization.losses import MaskL1Loss


class DiceLossTest(TestCase):
    def setUp(self):
        self.loss_obj = DiceLoss()

    def test_loss(self):
        y_true = np.array([1.0, 1.0, 0.0, 0.0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])
        mask = np.array([0.0, 1.0, 1.0, 0.0])
        weights = np.array([4.0, 5.0, 6.0, 7.0])
        loss = self.loss_obj(y_true, y_pred, mask, weights)
        self.assertAlmostEqual(loss.numpy(), 0.74358, delta=1e-4)

    def test_correct(self):
        y_true = np.array([1.0, 1.0, 0.0, 0.0])
        y_pred = y_true
        mask = np.array([0.0, 1.0, 1.0, 0.0])
        loss = self.loss_obj(y_true, y_pred, mask)
        self.assertAlmostEqual(loss.numpy(), 0.0, delta=1e-4)


class MaskL1LossTest(TestCase):
    def setUp(self):
        self.loss_obj = MaskL1Loss()

    def test_masked(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])
        mask = np.array([0.0, 1.0, 0.0, 1.0])
        loss = self.loss_obj(y_true, y_pred, mask)
        self.assertAlmostEqual(loss.numpy(), 2.7, delta=1e-4)


class DBLossTest(TestCase):
    def setUp(self):
        self.loss_obj = DBLoss()

    def test_loss(self):
        shrink_map = thresh_map = np.array(
            [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
        )
        p_map_pred = b_map_pred = t_map_pred = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        )
        shrink_mask = thresh_mask = np.ones_like(p_map_pred)
        y_true = np.stack(
            (shrink_map, shrink_mask, thresh_map, thresh_mask), axis=-1
        )
        y_pred = np.stack((p_map_pred, t_map_pred, b_map_pred), axis=-1)
        loss = self.loss_obj(y_true, y_pred)
        self.assertAlmostEqual(loss.numpy(), 14.1123, delta=1e-4)

    def test_correct(self):
        shrink_map = thresh_map = np.array(
            [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
        )
        p_map_pred, b_map_pred, t_map_pred = shrink_map, shrink_map, thresh_map
        shrink_mask = thresh_mask = np.ones_like(p_map_pred)
        y_true = np.stack(
            (shrink_map, shrink_mask, thresh_map, thresh_mask), axis=-1
        )
        y_pred = np.stack((p_map_pred, t_map_pred, b_map_pred), axis=-1)
        loss = self.loss_obj(y_true, y_pred)
        self.assertAlmostEqual(loss.numpy(), 0.0, delta=1e-4)
