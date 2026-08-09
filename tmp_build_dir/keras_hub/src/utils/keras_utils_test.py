import keras

from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.utils.keras_utils import clone_initializer


class CloneInitializerTest(TestCase):
    def test_config_equality(self):
        initializer = keras.initializers.VarianceScaling(
            scale=2.0,
            mode="fan_out",
        )
        clone = clone_initializer(initializer)
        self.assertAllEqual(initializer.get_config(), clone.get_config())

    def test_random_output(self):
        initializer = keras.initializers.VarianceScaling(
            scale=2.0,
            mode="fan_out",
        )
        clone = clone_initializer(initializer)
        self.assertNotAllEqual(initializer(shape=(2, 2)), clone(shape=(2, 2)))

    def test_strings(self):
        initializer = "glorot_uniform"
        clone = clone_initializer(initializer)
        self.assertAllEqual(initializer, clone)
