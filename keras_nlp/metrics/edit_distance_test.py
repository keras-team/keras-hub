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

        self.assertEqual(
            result, 0.0
        )

    def test_list_input(self):
        edit_distance = EditDistance(normalize=True)
        y_true = "the tiny little cat was found under the big funny bed".split()
        y_pred = "the cat was found under the bed".split()

        edit_distance_val = edit_distance(y_true, y_pred)
        self.assertAlmostEqual(
            edit_distance_val.numpy(),  0.364, delta=1e-3
        )

    def test_nested_list_input(self):
        edit_distance = EditDistance(normalize=True)
        y_true = ["the tiny little cat was found under the big funny bed".split(), "it is sunny today".split()]
        y_pred = ["the cat was found under the bed".split(), "it is sunny but with a hint of cloud cover".split()]

        edit_distance_val = edit_distance(y_true, y_pred)
        self.assertAlmostEqual(
            edit_distance_val.numpy(),  0.364, delta=1e-3
        )

    # def test_string_list_input(self):
    #     rouge = RougeL(use_stemmer=False)
    #     y_true = [
    #         "the tiny little cat was found under the big funny bed",
    #         "i really love contributing to KerasNLP",
    #     ]
    #     y_pred = [
    #         "the cat was under the bed",
    #         "i love contributing to KerasNLP",
    #     ]

    #     rouge_val = rouge(y_true, y_pred)
    #     self.assertDictAlmostEqual(
    #         rouge_val, {"precision": 1.0, "recall": 0.689, "f1_score": 0.807}
    #     )

    # def test_tensor_input(self):
    #     rouge = RougeL(use_stemmer=False)
    #     y_true = tf.constant(
    #         [
    #             "the tiny little cat was found under the big funny bed",
    #             "i really love contributing to KerasNLP",
    #         ]
    #     )
    #     y_pred = tf.constant(
    #         ["the cat was under the bed", "i love contributing to KerasNLP"]
    #     )

    #     rouge_val = rouge(y_true, y_pred)
    #     self.assertDictAlmostEqual(
    #         rouge_val, {"precision": 1.0, "recall": 0.689, "f1_score": 0.807}
    #     )

    # def test_rank_2_input(self):
    #     rouge = RougeL(use_stemmer=False)
    #     y_true = tf.constant(
    #         [
    #             ["the tiny little cat was found under the big funny bed"],
    #             ["i really love contributing to KerasNLP"],
    #         ]
    #     )
    #     y_pred = tf.constant(
    #         [["the cat was under the bed"], ["i love contributing to KerasNLP"]]
    #     )

    #     rouge_val = rouge(y_true, y_pred)
    #     self.assertDictAlmostEqual(
    #         rouge_val, {"precision": 1.0, "recall": 0.689, "f1_score": 0.807}
    #     )

    # def test_model_compile(self):
    #     inputs = keras.Input(shape=(), dtype="string")
    #     outputs = tf.strings.lower(inputs)
    #     model = keras.Model(inputs, outputs)

    #     model.compile(metrics=[RougeL()])

    #     x = tf.constant(["HELLO THIS IS FUN"])
    #     y = tf.constant(["hello this is awesome"])

    #     output = model.evaluate(x, y, return_dict=True)
    #     del output["loss"]
    #     self.assertDictAlmostEqual(
    #         output,
    #         {"precision": 0.75, "recall": 0.75, "f1_score": 0.75},
    #         typecast_to_numpy=False,
    #     )

    # def test_reset_state(self):
    #     rouge = RougeL()
    #     y_true = tf.constant(
    #         ["hey, this is great fun", "i love contributing to KerasNLP"]
    #     )
    #     y_pred = tf.constant(
    #         [
    #             "great fun indeed",
    #             "KerasNLP is awesome, i love contributing to it",
    #         ]
    #     )

    #     rouge.update_state(y_true, y_pred)
    #     rouge_val = rouge.result()
    #     self.assertDictAllValuesNotEqual(
    #         rouge_val, {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}
    #     )

    #     rouge.reset_state()
    #     rouge_val = rouge.result()
    #     self.assertDictEqual(
    #         rouge_val, {"precision": 0.0, "recall": 0.0, "f1_score": 0.0}
    #     )

    # def test_update_state(self):
    #     rouge = RougeL()
    #     y_true_1 = tf.constant(
    #         [
    #             "the tiny little cat was found under the big funny bed",
    #             "i really love contributing to KerasNLP",
    #         ]
    #     )
    #     y_pred_1 = tf.constant(
    #         ["the cat was under the bed", "i love contributing to KerasNLP"]
    #     )

    #     rouge.update_state(y_true_1, y_pred_1)
    #     rouge_val = rouge.result()
    #     self.assertDictAlmostEqual(
    #         rouge_val, {"precision": 1.0, "recall": 0.689, "f1_score": 0.807}
    #     )

    #     y_true_2 = tf.constant(["what is your favourite show"])
    #     y_pred_2 = tf.constant(["my favourite show is silicon valley"])

    #     rouge.update_state(y_true_2, y_pred_2)
    #     rouge_val = rouge.result()
    #     self.assertDictAlmostEqual(
    #         rouge_val, {"precision": 0.778, "recall": 0.593, "f1_score": 0.66}
    #     )

    # def test_merge_state(self):
    #     rouge_1 = RougeL()
    #     rouge_2 = RougeL()

    #     y_true_1 = tf.constant(
    #         [
    #             "the tiny little cat was found under the big funny bed",
    #             "i really love contributing to KerasNLP",
    #         ]
    #     )
    #     y_pred_1 = tf.constant(
    #         ["the cat was under the bed", "i love contributing to KerasNLP"]
    #     )

    #     y_true_2 = tf.constant(["what is your favourite show"])
    #     y_pred_2 = tf.constant(["my favourite show is silicon valley"])

    #     y_true_3 = tf.constant(["lorem ipsum dolor sit amet"])
    #     y_pred_3 = tf.constant(["lorem ipsum is simply dummy text"])

    #     rouge_1.update_state(y_true_1, y_pred_1)
    #     rouge_1.update_state(y_true_2, y_pred_2)
    #     rouge_val = rouge_1.result()
    #     self.assertDictAlmostEqual(
    #         rouge_val, {"precision": 0.778, "recall": 0.593, "f1_score": 0.66}
    #     )

    #     rouge_2.update_state(y_true_3, y_pred_3)
    #     rouge_val = rouge_2.result()
    #     self.assertDictAlmostEqual(
    #         rouge_val, {"precision": 0.333, "recall": 0.4, "f1_score": 0.364}
    #     )

    #     merged_rouge = RougeL()
    #     merged_rouge.merge_state([rouge_1, rouge_2])
    #     rouge_val = merged_rouge.result()
    #     self.assertDictAlmostEqual(
    #         rouge_val, {"precision": 0.667, "recall": 0.545, "f1_score": 0.586}
    #     )

    # def test_get_config(self):
    #     rouge = RougeL(
    #         use_stemmer=True,
    #         dtype=tf.float32,
    #         name="rouge_l_test",
    #     )

    #     config = rouge.get_config()
    #     expected_config_subset = {
    #         "use_stemmer": True,
    #     }
    #     self.assertEqual(config, {**config, **expected_config_subset})
