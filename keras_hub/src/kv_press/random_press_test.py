from keras_hub.src.kv_press.press_test import make_cache
from keras_hub.src.kv_press.random_press import RandomPress
from keras_hub.src.tests.test_case import TestCase


class RandomPressTest(TestCase):
    def test_same_seed_is_reproducible(self):
        cache = make_cache(2, 2, 8, 2, 4, lambda pos: pos)
        compressed_a = RandomPress(compression_ratio=0.5, seed=0).compress(
            cache
        )
        compressed_b = RandomPress(compression_ratio=0.5, seed=0).compress(
            cache
        )
        self.assertAllClose(compressed_a, compressed_b)

    def test_serialization(self):
        self.run_serialization_test(RandomPress(compression_ratio=0.7, seed=42))
