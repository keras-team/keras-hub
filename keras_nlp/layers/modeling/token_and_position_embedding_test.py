# Copyright 2023 The KerasNLP Authors
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

from keras_nlp.backend import ops
from keras_nlp.backend import random
from keras_nlp.layers.modeling.token_and_position_embedding import (
    TokenAndPositionEmbedding,
)
from keras_nlp.tests.test_case import TestCase


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
        self.assertAllEqual(outputs._keras_mask, mask)
