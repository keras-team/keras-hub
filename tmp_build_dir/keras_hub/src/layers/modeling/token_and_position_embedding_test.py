import numpy as np
from keras import ops
from keras import random
from keras.src.backend import get_keras_mask

from keras_hub.src.layers.modeling.token_and_position_embedding import (
    TokenAndPositionEmbedding,
)
from keras_hub.src.tests.test_case import TestCase


class TokenAndPositionEmbeddingTest(TestCase):
    def test_layer_behaviors(self):
        self.run_layer_test(
            cls=TokenAndPositionEmbedding,
            init_kwargs={
                "vocabulary_size": 5,
                "sequence_length": 4,
                "embedding_dim": 3,
                "embeddings_initializer": "ones",
            },
            input_data=random.randint(minval=0, maxval=5, shape=(2, 4)),
            expected_output_shape=(2, 4, 3),
            expected_output_data=ops.ones((2, 4, 3)) * 2,
            expected_num_trainable_weights=2,
        )

    def test_mask_propagation(self):
        test_layer = TokenAndPositionEmbedding(
            vocabulary_size=5,
            sequence_length=4,
            embedding_dim=3,
            mask_zero=True,
        )
        input_data = np.array([[1, 0], [1, 0]])
        mask = input_data != 0
        outputs = test_layer(input_data)
        self.assertAllEqual(get_keras_mask(outputs), mask)
