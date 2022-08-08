# Copyright 2022 The KerasNLP Authors
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

"""Tests for EditDistance."""

import tensorflow as tf
from tensorflow import keras

from keras_nlp.metrics import EditDistance


class EditDistanceTest(tf.test.TestCase):
    def test_initialization(self):
        edit_distance = EditDistance()
        result = edit_distance.result()

        self.assertEqual(result, 0.0)

    def test_1d_list_input_normalize(self):
        edit_distance = EditDistance()
        y_true = "the tiny little cat was found under the big funny bed".split()
        y_pred = "the cat was found under the bed".split()

        edit_distance_val = edit_distance(y_true, y_pred)
        self.assertAlmostEqual(edit_distance_val.numpy(), 0.364, delta=1e-3)

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
        self.assertAlmostEqual(edit_distance_val.numpy(), 0.733, delta=1e-3)

    def test_rank_1_tensor_input_normalize(self):
        edit_distance = EditDistance()
        y_true = tf.strings.split(
            "the tiny little cat was found under the big funny bed"
        )
        y_pred = tf.strings.split("the cat was found under the bed")

        edit_distance_val = edit_distance(y_true, y_pred)
        self.assertAlmostEqual(edit_distance_val.numpy(), 0.364, delta=1e-3)

    def test_rank_2_tensor_input_normalize(self):
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
        self.assertAlmostEqual(edit_distance_val.numpy(), 0.733, delta=1e-3)

    def test_rank_1_tensor_input_normalize_false(self):
        edit_distance = EditDistance(normalize=False)
        y_true = tf.strings.split(
            "the tiny little cat was found under the big funny bed"
        )
        y_pred = tf.strings.split("the cat was found under the bed")

        edit_distance_val = edit_distance(y_true, y_pred)
        self.assertAlmostEqual(edit_distance_val.numpy(), 4.0, delta=1e-3)

    def test_rank_2_tensor_input_normalize_false(self):
        edit_distance = EditDistance(normalize=False)
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
        self.assertAlmostEqual(edit_distance_val.numpy(), 5.5, delta=1e-3)

    def test_model_compile_normalize(self):
        inputs = keras.Input(shape=(None,), dtype="string")
        outputs = tf.strings.lower(inputs)
        model = keras.Model(inputs, outputs)

        model.compile(metrics=[EditDistance()])

        x = tf.strings.split(
            ["the tiny little cat was found under the big funny bed"]
        )
        y = tf.strings.split(["the cat was found under the bed"])

        output = model.evaluate(y, x, return_dict=True)

        self.assertAlmostEqual(output["edit_distance"], 0.364, delta=1e-3)

    def test_model_compile_normalize_false(self):
        inputs = keras.Input(shape=(None,), dtype="string")
        outputs = tf.strings.lower(inputs)
        model = keras.Model(inputs, outputs)

        model.compile(metrics=[EditDistance(normalize=False)])

        x = tf.strings.split(
            ["the tiny little cat was found under the big funny bed"]
        )
        y = tf.strings.split(["the cat was found under the bed"])

        output = model.evaluate(y, x, return_dict=True)

        self.assertAlmostEqual(output["edit_distance"], 4.0, delta=1e-3)

    def test_reset_state_normalize(self):
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

        edit_distance.update_state(y_true, y_pred)
        edit_distance_val = edit_distance.result()
        self.assertNotEqual(edit_distance_val.numpy(), 0.0)

        edit_distance.reset_state()
        edit_distance_val = edit_distance.result()
        self.assertEqual(edit_distance_val, 0.0)

    def test_update_state_normalize(self):
        edit_distance = EditDistance()
        y_true_1 = tf.strings.split(
            [
                "the tiny little cat was found under the big funny bed",
                "it is sunny today",
            ]
        )
        y_pred_1 = tf.strings.split(
            [
                "the cat was found under the bed",
                "it is sunny but with a hint of cloud cover",
            ]
        )

        edit_distance.update_state(y_true_1, y_pred_1)
        edit_distance_val = edit_distance.result()
        self.assertAlmostEqual(edit_distance_val.numpy(), 0.733, delta=1e-3)

        y_true_2 = tf.strings.split(["what is your favourite show"])
        y_pred_2 = tf.strings.split(["my favourite show is silicon valley"])

        edit_distance.update_state(y_true_2, y_pred_2)
        edit_distance_val = edit_distance.result()
        self.assertAlmostEqual(edit_distance_val.numpy(), 0.85, delta=1e-3)

    def test_update_state_normalize_false(self):
        edit_distance = EditDistance(normalize=False)
        y_true_1 = tf.strings.split(
            [
                "the tiny little cat was found under the big funny bed",
                "it is sunny today",
            ]
        )
        y_pred_1 = tf.strings.split(
            [
                "the cat was found under the bed",
                "it is sunny but with a hint of cloud cover",
            ]
        )

        edit_distance.update_state(y_true_1, y_pred_1)
        edit_distance_val = edit_distance.result()
        self.assertAlmostEqual(edit_distance_val.numpy(), 5.5, delta=1e-3)

        y_true_2 = tf.strings.split(["what is your favourite show"])
        y_pred_2 = tf.strings.split(["my favourite show is silicon valley"])

        edit_distance.update_state(y_true_2, y_pred_2)
        edit_distance_val = edit_distance.result()
        self.assertAlmostEqual(edit_distance_val.numpy(), 5.667, delta=1e-3)

    def test_merge_state_normalize(self):
        edit_distance_1 = EditDistance()
        edit_distance_2 = EditDistance()

        y_true_1 = tf.strings.split(
            [
                "the tiny little cat was found under the big funny bed",
                "it is sunny today",
            ]
        )
        y_pred_1 = tf.strings.split(
            [
                "the cat was found under the bed",
                "it is sunny but with a hint of cloud cover",
            ]
        )

        y_true_2 = tf.strings.split(["what is your favourite show"])
        y_pred_2 = tf.strings.split(["my favourite show is silicon valley"])

        y_true_3 = tf.strings.split(["lorem ipsum dolor sit amet"])
        y_pred_3 = tf.strings.split(["lorem ipsum is simply dummy text"])

        edit_distance_1.update_state(y_true_1, y_pred_1)
        edit_distance_1.update_state(y_true_2, y_pred_2)
        edit_distance_val = edit_distance_1.result()
        self.assertAlmostEqual(edit_distance_val.numpy(), 0.85, delta=1e-3)

        edit_distance_2.update_state(y_true_3, y_pred_3)
        edit_distance_val = edit_distance_2.result()
        self.assertAlmostEqual(edit_distance_val.numpy(), 0.8, delta=1e-3)

        merged_edit_distance = EditDistance()
        merged_edit_distance.merge_state([edit_distance_1, edit_distance_2])
        edit_distance_val = merged_edit_distance.result()
        self.assertAlmostEqual(edit_distance_val.numpy(), 0.84, delta=1e-3)

    def test_merge_state_normalize_false(self):
        edit_distance_1 = EditDistance(normalize=False)
        edit_distance_2 = EditDistance(normalize=False)

        y_true_1 = tf.strings.split(
            [
                "the tiny little cat was found under the big funny bed",
                "it is sunny today",
            ]
        )
        y_pred_1 = tf.strings.split(
            [
                "the cat was found under the bed",
                "it is sunny but with a hint of cloud cover",
            ]
        )

        y_true_2 = tf.strings.split(["what is your favourite show"])
        y_pred_2 = tf.strings.split(["my favourite show is silicon valley"])

        y_true_3 = tf.strings.split(["lorem ipsum dolor sit amet"])
        y_pred_3 = tf.strings.split(["lorem ipsum is simply dummy text"])

        edit_distance_1.update_state(y_true_1, y_pred_1)
        edit_distance_1.update_state(y_true_2, y_pred_2)
        edit_distance_val = edit_distance_1.result()
        self.assertAlmostEqual(edit_distance_val.numpy(), 5.667, delta=1e-3)

        edit_distance_2.update_state(y_true_3, y_pred_3)
        edit_distance_val = edit_distance_2.result()
        self.assertAlmostEqual(edit_distance_val.numpy(), 4.0, delta=1e-3)

        merged_edit_distance = EditDistance(normalize=False)
        merged_edit_distance.merge_state([edit_distance_1, edit_distance_2])
        edit_distance_val = merged_edit_distance.result()
        self.assertAlmostEqual(edit_distance_val.numpy(), 5.25, delta=1e-3)

    def test_get_config(self):
        rouge = EditDistance(
            normalize=False,
            dtype=tf.float32,
            name="edit_distance_test",
        )

        config = rouge.get_config()
        expected_config_subset = {
            "normalize": False,
        }
        self.assertEqual(config, {**config, **expected_config_subset})
