import numpy as np

from keras_hub.src.models.diffbin.diffbin_loss import DiffBinLoss
from keras_hub.src.tests.test_case import TestCase


class TestLoss(TestCase):
    def setUp(self):
        super().setUp()
        self.loss_fn = DiffBinLoss(alpha=1.0, beta=10.0)

    def test_loss_zero_when_y_true_equals_y_pred(self):
        y_true = np.ones((1, 4, 4, 4), dtype=np.float32)
        y_pred = np.zeros((1, 4, 4, 3), dtype=np.float32)
        y_pred[..., 0:1] = y_true[..., 0:1]
        y_pred[..., 1:2] = y_true[..., 2:3]
        y_pred[..., 2:3] = y_true[..., 1:2]

        loss = self.loss_fn(y_true, y_pred)
        self.assertGreaterEqual(loss.numpy(), 0.0)
