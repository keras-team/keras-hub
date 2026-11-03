import numpy as np
from keras import ops

from keras_hub.src.kv_press.knorm_press import KnormPress
from keras_hub.src.kv_press.press_test import make_cache
from keras_hub.src.tests.test_case import TestCase


class KnormPressTest(TestCase):
    def test_knorm_press_keeps_lowest_norm_keys(self):
        # Key norm grows with position: pos -> [pos + 1, 0, 0].
        seq_len = 5
        cache = make_cache(1, 1, seq_len, 1, 3, lambda pos: [pos + 1, 0, 0])
        # keep_len = round(5 * (1 - 0.4)) = 3, so positions 0, 1, 2 (norms
        # 1, 2, 3) should survive over 3, 4 (norms 4, 5).
        press = KnormPress(compression_ratio=0.4)
        compressed = press.compress(cache)
        kept_keys = ops.convert_to_numpy(compressed[0, 0, 0, :, 0, :])
        expected = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]])
        self.assertAllClose(kept_keys, expected)

    def test_serialization(self):
        self.run_serialization_test(KnormPress(compression_ratio=0.7))
