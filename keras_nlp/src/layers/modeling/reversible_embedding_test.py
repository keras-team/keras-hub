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

import keras
import numpy as np
from absl.testing import parameterized
from keras import ops
from keras import random

from keras_nlp.src.layers.modeling.reversible_embedding import (
    ReversibleEmbedding,
)
from keras_nlp.src.tests.test_case import TestCase
from keras_nlp.src.utils.keras_utils import has_quantization_support


class ReversibleEmbeddingTest(TestCase):
    @parameterized.named_parameters(
        ("tie_weights", True),
        ("untie_weights", False),
    )
    def test_layer_behaviors_tied(self, tie_weights):
        self.run_layer_test(
            cls=ReversibleEmbedding,
            init_kwargs={
                "input_dim": 100,
                "output_dim": 32,
                "tie_weights": tie_weights,
                "embeddings_initializer": "HeNormal",
                "logit_soft_cap": 50,
            },
            input_data=random.randint(minval=0, maxval=100, shape=(4, 10)),
            expected_output_shape=(4, 10, 32),
            expected_num_trainable_weights=1 if tie_weights else 2,
        )

    @parameterized.named_parameters(
        ("tie_weights", True),
        ("untie_weights", False),
    )
    def test_saving(self, tie_weights):
        input_data = random.randint(minval=0, maxval=100, shape=(4, 10))
        model = keras.Sequential(
            [
                ReversibleEmbedding(
                    input_dim=100,
                    output_dim=32,
                    tie_weights=tie_weights,
                )
            ]
        )
        path = os.path.join(self.get_temp_dir(), "model.keras")
        model_output = model(input_data)
        model.save(path, save_format="keras_v3")
        restored_model = keras.models.load_model(path)
        restored_output = restored_model(input_data)
        self.assertAllClose(model_output, restored_output)

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

        layer = ReversibleEmbedding(input_dim=3, output_dim=2, logit_soft_cap=5)
        layer.build()
        layer.embeddings.assign(np.array([[0.0, 0.0], [2.0, 2.0], [3.0, 3.0]]))
        out = layer(np.array(([[1.0, 1.0]])), reverse=True)
        self.assertAllClose(out, np.array([[0.0, 3.320184, 4.168273]]))

    def test_reverse_dtype(self):
        embedding = ReversibleEmbedding(100, 16, reverse_dtype="float32")
        input_data = ops.ones(shape=(4, 10, 16))
        output_data = embedding(input_data, reverse=True)
        self.assertEqual(output_data.shape, (4, 10, 100))
        self.assertDTypeEqual(output_data, "float32")

        if keras.config.backend() == "torch":
            import torch

            if not torch.cuda.is_available():
                self.skipTest("Torch CPU does not support float16")

        embedding = ReversibleEmbedding(100, 16, reverse_dtype="float16")
        input_data = ops.ones(shape=(4, 10, 16))
        output_data = embedding(input_data, reverse=True)
        self.assertEqual(output_data.shape, (4, 10, 100))
        self.assertDTypeEqual(output_data, "float16")

    @parameterized.named_parameters(
        ("tie_weights", True), ("untie_weights", False)
    )
    def test_quantize_int8(self, tie_weights):
        if not has_quantization_support():
            self.skipTest("This version of Keras doesn't support quantization.")

        layer_config = dict(
            input_dim=100, output_dim=32, tie_weights=tie_weights
        )
        layer = ReversibleEmbedding(**layer_config)
        layer.build()
        x = random.randint(shape=(64, 100), minval=0, maxval=9)
        x_reverse = random.uniform(shape=(64, 32))
        y_float = layer(x)
        y_reverse_float = layer(x_reverse, reverse=True)
        layer.quantize("int8")

        # Verify weights dtype
        if not tie_weights:
            self.assertEqual(
                keras.backend.standardize_dtype(layer.reverse_embeddings.dtype),
                "int8",
            )
            self.assertEqual(
                keras.backend.standardize_dtype(
                    layer.reverse_embeddings_scale.dtype
                ),
                layer.variable_dtype,
            )

        # Try eager call and verify output correctness
        y_quantized = layer(x)
        y_reverse_quantized = layer(x_reverse, reverse=True)
        mse = ops.mean(ops.square(y_float - y_quantized))
        mse_reverse = ops.mean(
            ops.square(y_reverse_float - y_reverse_quantized)
        )
        self.assertLess(mse, 1e-3)  # A weak correctness test
        self.assertLess(mse_reverse, 1e-3)  # A weak correctness test

        # Try saving and reloading the model
        model = keras.models.Sequential([layer])
        temp_filepath = os.path.join(
            self.get_temp_dir(), "quantized_model.keras"
        )
        model.save(temp_filepath)
        new_model = keras.models.load_model(temp_filepath)
        self.assertAllClose(model.predict(x), new_model.predict(x))

    @parameterized.named_parameters(
        ("tie_weights", True),
        ("untie_weights", False),
    )
    def test_quantize_dtype_argument(self, tie_weights):
        if not has_quantization_support():
            self.skipTest("This version of Keras doesn't support quantization.")

        self.run_layer_test(
            cls=ReversibleEmbedding,
            init_kwargs={
                "input_dim": 100,
                "output_dim": 32,
                "tie_weights": tie_weights,
                "embeddings_initializer": "HeNormal",
                "dtype": "int8_from_float32",
            },
            input_data=random.randint(minval=0, maxval=100, shape=(4, 10)),
            expected_output_shape=(4, 10, 32),
            expected_num_trainable_weights=0,
            expected_num_non_trainable_weights=2 if tie_weights else 4,
            expected_num_non_trainable_variables=2 if tie_weights else 4,
        )
