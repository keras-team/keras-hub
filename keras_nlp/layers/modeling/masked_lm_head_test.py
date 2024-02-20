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

from keras_nlp.backend import random
from keras_nlp.layers.modeling.masked_lm_head import MaskedLMHead
from keras_nlp.layers.modeling.reversible_embedding import ReversibleEmbedding
from keras_nlp.tests.test_case import TestCase


class MaskedLMHeadTest(TestCase):
    def test_layer_behaviors(self):
        self.run_layer_test(
            cls=MaskedLMHead,
            init_kwargs={
                "vocabulary_size": 100,
                "activation": "softmax",
                "kernel_initializer": "HeNormal",
                "bias_initializer": "Zeros",
            },
            input_data={
                "inputs": random.uniform(shape=(4, 10, 16)),
                "mask_positions": random.randint(
                    minval=0, maxval=10, shape=(4, 5)
                ),
            },
            expected_output_shape=(4, 5, 100),
            expected_num_trainable_weights=6,
        )

    def test_layer_behaviors_with_embedding(self):
        embedding = ReversibleEmbedding(100, 16)
        embedding.build((4, 10))
        self.run_layer_test(
            cls=MaskedLMHead,
            init_kwargs={
                "vocabulary_size": 100,
                "activation": "softmax",
                "kernel_initializer": "HeNormal",
                "bias_initializer": "Zeros",
                "token_embedding": embedding,
            },
            input_data={
                "inputs": random.uniform(shape=(4, 10, 16)),
                "mask_positions": random.randint(
                    minval=0, maxval=10, shape=(4, 5)
                ),
            },
            expected_output_shape=(4, 5, 100),
            expected_num_trainable_weights=6,
            run_precision_checks=False,
        )

    def test_value_error_when_neither_embedding_or_vocab_size_set(self):
        with self.assertRaises(ValueError):
            MaskedLMHead()

    def test_value_error_when_vocab_size_mismatch(self):
        embedding = ReversibleEmbedding(100, 16)
        embedding.build((4, 10))
        with self.assertRaises(ValueError):
            MaskedLMHead(
                vocabulary_size=101,
                token_embedding=embedding,
            )
