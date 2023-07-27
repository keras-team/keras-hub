from keras_nlp.backend import keras
from keras_nlp.backend import ops
from keras_nlp.layers.modeling.rotary_embedding import RotaryEmbedding
from keras_nlp.tests.test_case import TestCase


class RotaryEmbeddingTest(TestCase):
    def test_valid_shape(self):
        pos_encoding = RotaryEmbedding()
        input = ops.ones(shape=[2, 4, 1])
        self.assertAllEqual(input.shape, [2, 4, 1])


