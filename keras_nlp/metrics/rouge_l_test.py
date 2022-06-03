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

"""Tests for RougeL."""

import tensorflow as tf
from tensorflow import keras

from keras_nlp.metrics import RougeL


class RougeLTest(tf.test.TestCase):
    def test_initialization(self):
        rouge = RougeL()
        self.assertEqual(rouge.result().numpy(), 0.0)

    def test_string_input(self):
        rouge = RougeL(use_stemmer=False)
        y_true = "the tiny little cat was found under the big funny bed"
        y_pred = "the cat was under the bed"

        rouge_val = rouge(y_true, y_pred)
        self.assertAlmostEqual(rouge_val.numpy(), 0.706, delta=1e-3)

    def test_string_list_input(self):
        rouge = RougeL(use_stemmer=False)
        y_true = [
            "the tiny little cat was found under the big funny bed",
            "i really love contributing to KerasNLP",
        ]
        y_pred = [
            "the cat was under the bed",
            "i love contributing to KerasNLP",
        ]

        rouge_val = rouge(y_true, y_pred)
        self.assertAlmostEqual(rouge_val.numpy(), 0.807, delta=1e-3)

    def test_tensor_input(self):
        rouge = RougeL(use_stemmer=False)
        y_true = tf.constant(
            [
                "the tiny little cat was found under the big funny bed",
                "i really love contributing to KerasNLP",
            ]
        )
        y_pred = tf.constant(
            ["the cat was under the bed", "i love contributing to KerasNLP"]
        )

        rouge_val = rouge(y_true, y_pred)
        self.assertAlmostEqual(rouge_val.numpy(), 0.807, delta=1e-3)

    def test_rank_2_input(self):
        rouge = RougeL(use_stemmer=False)
        y_true = tf.constant(
            [
                ["the tiny little cat was found under the big funny bed"],
                ["i really love contributing to KerasNLP"],
            ]
        )
        y_pred = tf.constant(
            [["the cat was under the bed"], ["i love contributing to KerasNLP"]]
        )

        rouge_val = rouge(y_true, y_pred)
        self.assertAlmostEqual(rouge_val.numpy(), 0.807, delta=1e-3)

    def model_compile(self):
        inputs = keras.Input(shape=(), dtype="string")
        outputs = tf.strings.lower(inputs)
        model = keras.Model(inputs, outputs)

        model.compile(metrics=[RougeL()])

        x = tf.constant(["HELLO THIS IS FUN"])
        y = tf.constant(["hello this is awesome"])

        output = model.evaluate(x, y, return_dict=True)
        self.assertAlmostEqual(output["rouge-l"], 0.75, delta=1e-3)

    def test_precision(self):
        rouge = RougeL(metric_type="precision", use_stemmer=False)
        y_true = tf.constant(
            [
                "the tiny little cat was found under the big funny bed",
                "i really love contributing to KerasNLP",
            ]
        )
        y_pred = tf.constant(
            ["the cat was under the bed", "i love contributing to KerasNLP"]
        )

        rouge_val = rouge(y_true, y_pred)
        self.assertAlmostEqual(rouge_val.numpy(), 1, delta=1e-3)

    def test_recall(self):
        rouge = RougeL(metric_type="recall", use_stemmer=False)
        y_true = tf.constant(
            [
                "the tiny little cat was found under the big funny bed",
                "i really love contributing to KerasNLP",
            ]
        )
        y_pred = tf.constant(
            ["the cat was under the bed", "i love contributing to KerasNLP"]
        )

        rouge_val = rouge(y_true, y_pred)
        self.assertAlmostEqual(rouge_val.numpy(), 0.689, delta=1e-3)

    def test_reset_state(self):
        rouge = RougeL()
        y_true = tf.constant(
            ["hey, this is great fun", "i love contributing to KerasNLP"]
        )
        y_pred = tf.constant(
            [
                "great fun indeed",
                "KerasNLP is awesome, i love contributing to it",
            ]
        )

        rouge.update_state(y_true, y_pred)
        self.assertNotEqual(rouge.result(), 0.0)

        rouge.reset_state()
        self.assertEqual(rouge.result(), 0.0)

    def test_update_state(self):
        rouge = RougeL()
        y_true_1 = tf.constant(
            [
                "the tiny little cat was found under the big funny bed",
                "i really love contributing to KerasNLP",
            ]
        )
        y_pred_1 = tf.constant(
            ["the cat was under the bed", "i love contributing to KerasNLP"]
        )

        rouge.update_state(y_true_1, y_pred_1)
        rouge_val = rouge.result()
        self.assertAlmostEqual(rouge_val.numpy(), 0.807, delta=1e-3)

        y_true_2 = tf.constant(["what is your favourite show"])
        y_pred_2 = tf.constant(["my favourite show is silicon valley"])

        rouge.update_state(y_true_2, y_pred_2)
        rouge_val = rouge.result()
        self.assertAlmostEqual(rouge_val.numpy(), 0.659, delta=1e-3)

    def test_merge_state(self):
        rouge_1 = RougeL()
        rouge_2 = RougeL()

        y_true_1 = tf.constant(
            [
                "the tiny little cat was found under the big funny bed",
                "i really love contributing to KerasNLP",
            ]
        )
        y_pred_1 = tf.constant(
            ["the cat was under the bed", "i love contributing to KerasNLP"]
        )

        y_true_2 = tf.constant(["what is your favourite show"])
        y_pred_2 = tf.constant(["my favourite show is silicon valley"])

        y_true_3 = tf.constant(["lorem ipsum dolor sit amet"])
        y_pred_3 = tf.constant(["lorem ipsum is simply dummy text"])

        rouge_1.update_state(y_true_1, y_pred_1)
        rouge_1.update_state(y_true_2, y_pred_2)
        self.assertAlmostEqual(rouge_1.result().numpy(), 0.659, delta=1e-3)

        rouge_2.update_state(y_true_3, y_pred_3)
        self.assertAlmostEqual(rouge_2.result().numpy(), 0.364, delta=1e-3)

        merged_rouge = RougeL()
        merged_rouge.merge_state([rouge_1, rouge_2])
        self.assertAlmostEqual(merged_rouge.result().numpy(), 0.586, delta=1e-3)

    def test_get_config(self):
        rouge = RougeL(
            metric_type="precision",
            use_stemmer=True,
            dtype=tf.float32,
            name="rouge_l_test",
        )

        config = rouge.get_config()
        expected_config_subset = {
            "metric_type": "precision",
            "use_stemmer": True,
        }
        self.assertEqual(config, {**config, **expected_config_subset})
