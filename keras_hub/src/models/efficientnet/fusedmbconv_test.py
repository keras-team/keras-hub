import keras

from keras_hub.src.models.efficientnet.fusedmbconv import FusedMBConvBlock
from keras_hub.src.tests.test_case import TestCase


class FusedMBConvBlockTest(TestCase):
    def test_same_input_output_shapes(self):
        inputs = keras.random.normal(shape=(1, 64, 64, 32), dtype="float32")
        layer = FusedMBConvBlock(input_filters=32, output_filters=32)

        output = layer(inputs)
        self.assertEquals(output.shape, (1, 64, 64, 32))
        self.assertLen(output, 1)

    def test_different_input_output_shapes(self):
        inputs = keras.random.normal(shape=(1, 64, 64, 32), dtype="float32")
        layer = FusedMBConvBlock(input_filters=32, output_filters=48)

        output = layer(inputs)
        self.assertEquals(output.shape, (1, 64, 64, 48))
        self.assertLen(output, 1)

    def test_squeeze_excitation_ratio(self):
        inputs = keras.random.normal(shape=(1, 64, 64, 32), dtype="float32")
        layer = FusedMBConvBlock(
            input_filters=32, output_filters=48, se_ratio=0.25
        )

        output = layer(inputs)
        self.assertEquals(output.shape, (1, 64, 64, 48))
        self.assertLen(output, 1)
