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
import os

import numpy as np
from absl.testing import parameterized

from keras_nlp.backend import keras
from keras_nlp.backend import ops
from keras_nlp.layers.modeling.reversible_embedding import ReversibleEmbedding
from keras_nlp.tests.test_case import TestCase


class ReversibleEmbeddingTest(TestCase):
    @parameterized.named_parameters(
        ("tie_weights", True),
        ("untie_weights", False),
    )
    def test_valid_call(self, tie_weights):
        embedding = ReversibleEmbedding(100, 32, tie_weights=tie_weights)
        inputs = keras.Input(shape=(10,))
        hidden_states = embedding(inputs)
        outputs = embedding(hidden_states, reverse=True)
        model = keras.Model(inputs, outputs)

        input_data = ops.random.uniform(shape=(4, 10))
        model(input_data)

    def test_correctness(self):
        layer = ReversibleEmbedding(input_dim=3, output_dim=2)
        layer.build()
        layer.embeddings.assign(np.array([[0.0, 0.0], [2.0, 2.0], [3.0, 3.0]]))
        out = layer(np.array(([2, 1, 0])))
        self.assertAllClose(out, np.array([[3.0, 3.0], [2.0, 2.0], [0.0, 0.0]]))

        layer = ReversibleEmbedding(input_dim=3, output_dim=2)
        layer.build()
        layer.embeddings.assign(np.array([[0.0, 0.0], [2.0, 2.0], [3.0, 3.0]]))
        out = layer(np.array(([[1.0, 1.0]])), reverse=True)
        self.assertAllClose(out, np.array([[0.0, 4.0, 6.0]]))

    def test_config(self):
        original = ReversibleEmbedding(
            100,
            32,
            tie_weights=False,
            embeddings_initializer="HeNormal",
        )
        restored = ReversibleEmbedding.from_config(original.get_config())
        restored.set_weights(original.get_weights())
        self.assertEqual(restored.get_config(), original.get_config())

    def test_tied_checkpoint_untied_weights(self):
        embedding = ReversibleEmbedding(100, 16, tie_weights=True)
        inputs = keras.Input(shape=(10,), dtype="int32")
        hidden_states = embedding(inputs)
        outputs = embedding(hidden_states, reverse=True)
        tied_model = keras.Model(inputs, outputs)
        path = os.path.join(self.get_temp_dir(), "checkpoint.weights.h5")
        tied_model.save_weights(path)

        embedding = ReversibleEmbedding(100, 16, tie_weights=False)
        inputs = keras.Input(shape=(10,), dtype="int32")
        hidden_states = embedding(inputs)
        outputs = embedding(hidden_states, reverse=True)
        untied_model = keras.Model(inputs, outputs)
        untied_model.load_weights(path)

        input_data = ops.ones(shape=(4, 10), dtype="int32")
        self.assertAllClose(untied_model(input_data), tied_model(input_data))

    @parameterized.named_parameters(
        ("tie_weights", True),
        ("untie_weights", False),
    )
    def test_saved_model(self, tie_weights):
        embedding = ReversibleEmbedding(100, 16, tie_weights=tie_weights)
        inputs = keras.Input(shape=(10,), dtype="int32")
        hidden_states = embedding(inputs)
        outputs = embedding(hidden_states, reverse=True)
        model = keras.Model(inputs, outputs)

        input_data = ops.ones(shape=(4, 10), dtype="int32")
        model_output = model(input_data)
        path = os.path.join(self.get_temp_dir(), "model.keras")
        model.save(path, save_format="keras_v3")
        restored_model = keras.models.load_model(path)

        restored_output = restored_model(input_data)
        self.assertAllClose(model_output, restored_output)
