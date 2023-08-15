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

import pytest
import tensorflow as tf

from keras_nlp.backend import keras
from keras_nlp.metrics.edit_distance import EditDistance
from keras_nlp.tests.test_case import TestCase


class EditDistanceTest(TestCase):
    def test_initialization(self):
        edit_distance = EditDistance()
        result = edit_distance.result()

        self.assertEqual(result, 0.0)

    def test_1d_list_input_normalize(self):
        edit_distance = EditDistance()
        y_true = "the tiny little cat was found under the big funny bed".split()
        y_pred = "the cat was found under the bed".split()

        edit_distance_val = edit_distance(y_true, y_pred)
        self.assertAlmostEqual(edit_distance_val, 0.364, delta=1e-3)

    def test_2d_list_input_normalize(self):
        edit_distance = EditDistance()
        y_true = [
            "the tiny little cat was found under the big funny bed".split(),
            "it is sunny today".split(),
        ]
        y_pred = [
            "the cat was found under the bed".split(),
            "it is sunny but with a hint of cloud cover".split(),
        ]

        edit_distance_val = edit_distance(y_true, y_pred)
        self.assertAlmostEqual(edit_distance_val, 0.733, delta=1e-3)

    def test_1d_list_input_normalize_false(self):
        edit_distance = EditDistance(normalize=False)
        y_true = "the tiny little cat was found under the big funny bed".split()
        y_pred = "the cat was found under the bed".split()

        edit_distance_val = edit_distance(y_true, y_pred)
        self.assertAlmostEqual(edit_distance_val, 4.0, delta=1e-3)

    def test_2d_list_input_normalize_false(self):
        edit_distance = EditDistance(normalize=False)
        y_true = [
            "the tiny little cat was found under the big funny bed".split(),
            "it is sunny today".split(),
        ]
        y_pred = [
            "the cat was found under the bed".split(),
            "it is sunny but with a hint of cloud cover".split(),
        ]

        edit_distance_val = edit_distance(y_true, y_pred)
        self.assertAlmostEqual(edit_distance_val, 5.5, delta=1e-3)

    def test_tensor_input(self):
        edit_distance = EditDistance()
        y_true = tf.strings.split(
            [
                "the tiny little cat was found under the big funny bed",
                "it is sunny today",
            ]
        )
        y_pred = tf.strings.split(
            [
                "the cat was found under the bed",
                "it is sunny but with a hint of cloud cover",
            ]
        )

        edit_distance_val = edit_distance(y_true, y_pred)
        self.assertAlmostEqual(edit_distance_val, 0.733, delta=1e-3)

    @pytest.mark.tf_only  # string model output only applies to tf.
    def test_model_compile_normalize(self):
        inputs = keras.Input(shape=(None,), dtype="string")
        outputs = keras.layers.Identity()(inputs)
        model = keras.Model(inputs, outputs)

        model.compile(metrics=[EditDistance()])

        y_pred = x = tf.strings.split(["the cat was found under the bed"])
        y = tf.strings.split(
            ["the tiny little cat was found under the big funny bed"]
        )

        output = model.compute_metrics(x, y, y_pred, sample_weight=None)
        self.assertAlmostEqual(output["edit_distance"], 0.364, delta=1e-3)

    @pytest.mark.tf_only  # string model output only applies to tf.
    def test_model_compile_normalize_false(self):
        inputs = keras.Input(shape=(None,), dtype="string")
        outputs = keras.layers.Identity()(inputs)
        model = keras.Model(inputs, outputs)

        model.compile(metrics=[EditDistance(normalize=False)])

        y_pred = x = tf.strings.split(["the cat was found under the bed"])
        y = tf.strings.split(
            ["the tiny little cat was found under the big funny bed"]
        )

        output = model.compute_metrics(x, y, y_pred, sample_weight=None)
        self.assertAlmostEqual(output["edit_distance"], 4.0, delta=1e-3)

    def test_rank_1_tensor_input_normalize(self):
        edit_distance = EditDistance()
        y_true = tf.strings.split(
            "the tiny little cat was found under the big funny bed"
        )
        y_pred = tf.strings.split("the cat was found under the bed")

        edit_distance_val = edit_distance(y_true, y_pred)
        self.assertAlmostEqual(edit_distance_val, 0.364, delta=1e-3)

    def test_reset_state_normalize(self):
        edit_distance = EditDistance()
        y_true = [
            "the tiny little cat was found under the big funny bed".split(),
            "it is sunny today".split(),
        ]
        y_pred = [
            "the cat was found under the bed".split(),
            "it is sunny but with a hint of cloud cover".split(),
        ]

        edit_distance.update_state(y_true, y_pred)
        edit_distance_val = edit_distance.result()
        self.assertNotEqual(edit_distance_val, 0.0)

        edit_distance.reset_state()
        edit_distance_val = edit_distance.result()
        self.assertEqual(edit_distance_val, 0.0)

    def test_update_state_normalize(self):
        edit_distance = EditDistance()
        y_true_1 = [
            "the tiny little cat was found under the big funny bed".split(),
            "it is sunny today".split(),
        ]
        y_pred_1 = [
            "the cat was found under the bed".split(),
            "it is sunny but with a hint of cloud cover".split(),
        ]

        edit_distance.update_state(y_true_1, y_pred_1)
        edit_distance_val = edit_distance.result()
        self.assertAlmostEqual(edit_distance_val, 0.733, delta=1e-3)

        y_true_2 = tf.strings.split(["what is your favourite show"])
        y_pred_2 = tf.strings.split(["my favourite show is silicon valley"])

        edit_distance.update_state(y_true_2, y_pred_2)
        edit_distance_val = edit_distance.result()
        self.assertAlmostEqual(edit_distance_val, 0.85, delta=1e-3)

    def test_update_state_normalize_false(self):
        edit_distance = EditDistance(normalize=False)
        y_true_1 = [
            "the tiny little cat was found under the big funny bed".split(),
            "it is sunny today".split(),
        ]
        y_pred_1 = [
            "the cat was found under the bed".split(),
            "it is sunny but with a hint of cloud cover".split(),
        ]

        edit_distance.update_state(y_true_1, y_pred_1)
        edit_distance_val = edit_distance.result()
        self.assertAlmostEqual(edit_distance_val, 5.5, delta=1e-3)

        y_true_2 = tf.strings.split(["what is your favourite show"])
        y_pred_2 = tf.strings.split(["my favourite show is silicon valley"])

        edit_distance.update_state(y_true_2, y_pred_2)
        edit_distance_val = edit_distance.result()
        self.assertAlmostEqual(edit_distance_val, 5.667, delta=1e-3)

    def test_get_config(self):
        rouge = EditDistance(
            normalize=False,
            dtype="float32",
            name="edit_distance_test",
        )

        config = rouge.get_config()
        expected_config_subset = {
            "normalize": False,
        }
        self.assertEqual(config, {**config, **expected_config_subset})
