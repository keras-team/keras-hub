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

"""Tests for RougeN."""
import tensorflow as tf

from keras_nlp.backend import keras
from keras_nlp.metrics.rouge_n import RougeN
from keras_nlp.tests.test_case import TestCase


class RougeNTest(TestCase):
    def test_initialization(self):
        rouge = RougeN()
        result = rouge.result()

        self.assertAllClose(
            result,
            {"precision": 0.0, "recall": 0.0, "f1_score": 0.0},
        )

    def test_string_input(self):
        rouge = RougeN(order=2, use_stemmer=False)
        y_true = "the tiny little cat was found under the big funny bed"
        y_pred = "the cat was under the bed"

        rouge_val = rouge(y_true, y_pred)
        self.assertAllClose(
            rouge_val,
            {"precision": 0.4, "recall": 0.2, "f1_score": 0.266666},
        )

    def test_string_list_input(self):
        rouge = RougeN(order=2, use_stemmer=False)
        y_true = [
            "the tiny little cat was found under the big funny bed",
            "i really love contributing to KerasNLP",
        ]
        y_pred = [
            "the cat was under the bed",
            "i love contributing to KerasNLP",
        ]

        rouge_val = rouge(y_true, y_pred)
        self.assertAllClose(
            rouge_val,
            {"precision": 0.575, "recall": 0.4, "f1_score": 0.466666},
        )

    def test_tensor_input(self):
        rouge = RougeN(order=2, use_stemmer=False)
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
        self.assertAllClose(
            rouge_val,
            {"precision": 0.575, "recall": 0.4, "f1_score": 0.466666},
        )

    def test_rank_2_input(self):
        rouge = RougeN(order=2, use_stemmer=False)
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
        self.assertAllClose(
            rouge_val,
            {"precision": 0.575, "recall": 0.4, "f1_score": 0.466666},
        )

    def test_model_compile(self):
        inputs = keras.Input(shape=(), dtype="string")
        outputs = tf.strings.lower(inputs)
        model = keras.Model(inputs, outputs)

        model.compile(metrics=[RougeN()])

        x = tf.constant(["HELLO THIS IS FUN"])
        y = tf.constant(["hello this is awesome"])

        output = model.evaluate(x, y, return_dict=True)
        del output["loss"]
        self.assertAllClose(
            output,
            {"precision": 0.666666, "recall": 0.666666, "f1_score": 0.666666},
        )

    def test_incorrect_order(self):
        with self.assertRaises(ValueError):
            _ = RougeN(order=10)

    def test_different_order(self):
        rouge = RougeN(order=3, use_stemmer=False)
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
        self.assertAllClose(
            rouge_val,
            {"precision": 0.333333, "recall": 0.25, "f1_score": 0.285714},
        )

    def test_reset_state(self):
        rouge = RougeN()
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
        rouge_val = rouge.result()
        self.assertNotAllClose(
            rouge_val,
            {"precision": 0.0, "recall": 0.0, "f1_score": 0.0},
        )

        rouge.reset_state()
        rouge_val = rouge.result()
        self.assertAllClose(
            rouge_val,
            {"precision": 0.0, "recall": 0.0, "f1_score": 0.0},
        )

    def test_update_state(self):
        rouge = RougeN()
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
        self.assertAllClose(
            rouge_val,
            {"precision": 0.575, "recall": 0.4, "f1_score": 0.466666},
        )

        y_true_2 = tf.constant(["what is your favourite show"])
        y_pred_2 = tf.constant(["my favourite show is silicon valley"])

        rouge.update_state(y_true_2, y_pred_2)
        rouge_val = rouge.result()
        self.assertAllClose(
            rouge_val,
            {"precision": 0.45, "recall": 0.35, "f1_score": 0.385185},
        )

    def test_get_config(self):
        rouge = RougeN(
            order=5,
            use_stemmer=True,
            dtype="float32",
            name="rouge_n_test",
        )

        config = rouge.get_config()
        expected_config_subset = {
            "order": 5,
            "use_stemmer": True,
        }

        self.assertEqual(config, {**config, **expected_config_subset})
