from keras import ops

from keras_hub.src.models.diffbin.losses import DiceLoss
from keras_hub.src.models.diffbin.losses import DiffBinLoss
from keras_hub.src.models.diffbin.losses import MaskL1Loss
from keras_hub.src.tests.test_case import TestCase


class DiceLossTest(TestCase):
    def setUp(self):
        self.loss_obj = DiceLoss()

    def test_loss(self):
        y_true = ops.array([1.0, 1.0, 0.0, 0.0])
        y_pred = ops.array([0.1, 0.2, 0.3, 0.4])
        mask = ops.array([0.0, 1.0, 1.0, 0.0])
        weights = ops.array([4.0, 5.0, 6.0, 7.0])
        loss = self.loss_obj(y_true, y_pred, mask, weights)
        self.assertAlmostEqual(loss, 0.74358, delta=1e-4)

    def test_correct(self):
        y_true = ops.array([1.0, 1.0, 0.0, 0.0])
        y_pred = y_true
        mask = ops.array([0.0, 1.0, 1.0, 0.0])
        loss = self.loss_obj(y_true, y_pred, mask)
        self.assertAlmostEqual(loss, 0.0, delta=1e-4)


class MaskL1LossTest(TestCase):
    def setUp(self):
        self.loss_obj = MaskL1Loss()

    def test_masked(self):
        y_true = ops.array([1.0, 2.0, 3.0, 4.0])
        y_pred = ops.array([0.1, 0.2, 0.3, 0.4])
        mask = ops.array([0.0, 1.0, 0.0, 1.0])
        loss = self.loss_obj(y_true, y_pred, mask)
        self.assertAlmostEqual(loss, 2.7, delta=1e-4)


class DiffBinLossTest(TestCase):
    def setUp(self):
        self.loss_obj = DiffBinLoss()

    def test_loss(self):
        shrink_map = thresh_map = ops.array(
            [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
        )
        p_map_pred = b_map_pred = t_map_pred = ops.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        )
        shrink_mask = thresh_mask = ops.ones_like(p_map_pred)
        y_true = ops.stack(
            (shrink_map, shrink_mask, thresh_map, thresh_mask), axis=-1
        )
        y_pred = ops.stack((p_map_pred, t_map_pred, b_map_pred), axis=-1)
        loss = self.loss_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 14.1123, delta=1e-4)

    def test_correct(self):
        shrink_map = thresh_map = ops.array(
            [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
        )
        p_map_pred, b_map_pred, t_map_pred = shrink_map, shrink_map, thresh_map
        shrink_mask = thresh_mask = ops.ones_like(p_map_pred)
        y_true = ops.stack(
            (shrink_map, shrink_mask, thresh_map, thresh_mask), axis=-1
        )
        y_pred = ops.stack((p_map_pred, t_map_pred, b_map_pred), axis=-1)
        loss = self.loss_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 0.0, delta=1e-4)
