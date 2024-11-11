import keras

from keras_hub.src.models.efficientnet.convbnact import ConvBNActBlock
from keras_hub.src.tests.test_case import TestCase


class ConvBNActBlockTest(TestCase):
    def test_same_input_output_shapes(self):
        inputs = keras.random.normal(shape=(1, 64, 64, 32), dtype="float32")
        layer = ConvBNActBlock(input_filters=32, output_filters=32)

        output = layer(inputs)
        self.assertEquals(output.shape, (1, 64, 64, 32))
        self.assertLen(output, 1)

    def test_different_input_output_shapes(self):
        inputs = keras.random.normal(shape=(1, 64, 64, 32), dtype="float32")
        layer = ConvBNActBlock(input_filters=32, output_filters=48)

        output = layer(inputs)
        self.assertEquals(output.shape, (1, 64, 64, 48))
        self.assertLen(output, 1)
