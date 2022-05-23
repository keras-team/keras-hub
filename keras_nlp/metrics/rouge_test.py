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

"""Tests for Rouge."""

import tensorflow as tf

from keras_nlp.metrics import Rouge


class RougeTest(tf.test.TestCase):
    def test_initialization(self):
        rouge = Rouge()
        self.assertEqual(rouge.result().numpy(), 0.0)

    def test_string_input(self):
        rouge = Rouge(
            variant="rouge2", metric_type="f1_score", use_stemmer=False
        )
        y_true = "hey, this is great fun"
        y_pred = "great fun indeed"

        rouge_val = rouge(y_true, y_pred)
        self.assertAlmostEqual(rouge_val.numpy(), 0.333, delta=1e-3)

    def test_string_list_input(self):
        rouge = Rouge(
            variant="rouge2", metric_type="f1_score", use_stemmer=False
        )
        y_true = ["hey, this is great fun", "i love contributing to KerasNLP"]
        y_pred = ["great fun indeed", "contributing to KerasNLP is delightful"]

        rouge_val = rouge(y_true, y_pred)
        self.assertAlmostEqual(rouge_val.numpy(), 0.417, delta=1e-3)

    def test_tensor_input(self):
        rouge = Rouge(
            variant="rouge2", metric_type="f1_score", use_stemmer=False
        )
        y_true = tf.constant(
            ["hey, this is great fun", "i love contributing to KerasNLP"]
        )
        y_pred = tf.constant(
            ["great fun indeed", "contributing to KerasNLP is delightful"]
        )

        rouge_val = rouge(y_true, y_pred)
        self.assertAlmostEqual(rouge_val.numpy(), 0.417, delta=1e-3)

    def test_rouge_l(self):
        rouge = Rouge(
            variant="rougeL", metric_type="f1_score", use_stemmer=False
        )
        y_true = tf.constant(
            ["hey, this is great fun", "i love contributing to KerasNLP"]
        )
        y_pred = tf.constant(
            ["great fun indeed", "contributing to KerasNLP is delightful"]
        )

        rouge_val = rouge(y_true, y_pred)
        self.assertAlmostEqual(rouge_val.numpy(), 0.55, delta=1e-3)

    def test_rouge_l_sum(self):
        rouge = Rouge(
            variant="rougeLsum", metric_type="f1_score", use_stemmer=False
        )
        y_true = tf.constant(
            ["hey, this is great fun", "i love contributing to KerasNLP"]
        )
        y_pred = tf.constant(
            ["great fun indeed", "contributing to KerasNLP is delightful"]
        )

        rouge_val = rouge(y_true, y_pred)
        self.assertAlmostEqual(rouge_val.numpy(), 0.55, delta=1e-3)

    def test_incorrect_variant(self):
        with self.assertRaises(ValueError):
            _ = Rouge(
                variant="rouge10", metric_type="f1_score", use_stemmer=False
            )

    def test_precision(self):
        rouge = Rouge(
            variant="rouge3", metric_type="precision", use_stemmer=False
        )
        y_true = tf.constant(
            ["hey, this is great fun", "i love contributing to KerasNLP"]
        )
        y_pred = tf.constant(
            [
                "great fun indeed",
                "KerasNLP is awesome, i love contributing to it",
            ]
        )

        rouge_val = rouge(y_true, y_pred)
        self.assertAlmostEqual(rouge_val.numpy(), 0.167, delta=1e-3)

    def test_recall(self):
        rouge = Rouge(variant="rouge3", metric_type="recall", use_stemmer=False)
        y_true = tf.constant(
            ["hey, this is great fun", "i love contributing to KerasNLP"]
        )
        y_pred = tf.constant(
            [
                "great fun indeed",
                "KerasNLP is awesome, i love contributing to it",
            ]
        )

        rouge_val = rouge(y_true, y_pred)
        self.assertAlmostEqual(rouge_val.numpy(), 0.333, delta=1e-3)

    def test_reset_state(self):
        rouge = Rouge()
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
        rouge = Rouge()
        y_true_1 = tf.constant(
            ["hey, this is great fun", "i love contributing to KerasNLP"]
        )
        y_pred_1 = tf.constant(
            [
                "great fun indeed",
                "KerasNLP is awesome, i love contributing to it",
            ]
        )

        rouge.update_state(y_true_1, y_pred_1)
        rouge_val = rouge.result()
        self.assertAlmostEqual(rouge_val.numpy(), 0.439, delta=1e-3)

        y_true_2 = tf.constant(["what is your favourite show"])
        y_pred_2 = tf.constant(["my favourite show is silicon valley"])

        rouge.update_state(y_true_2, y_pred_2)
        rouge_val = rouge.result()
        self.assertAlmostEqual(rouge_val.numpy(), 0.367, delta=1e-3)

    def test_merge_state(self):
        rouge_1 = Rouge()
        rouge_2 = Rouge()

        y_true_1 = tf.constant(
            ["hey, this is great fun", "i love contributing to KerasNLP"]
        )
        y_pred_1 = tf.constant(
            [
                "great fun indeed",
                "KerasNLP is awesome, i love contributing to it",
            ]
        )

        y_true_2 = tf.constant(["what is your favourite show"])
        y_pred_2 = tf.constant(["my favourite show is silicon valley"])

        y_true_3 = tf.constant(["lorem ipsum dolor sit amet"])
        y_pred_3 = tf.constant(["lorem ipsum is simply dummy text"])

        rouge_1.update_state(y_true_1, y_pred_1)
        rouge_1.update_state(y_true_2, y_pred_2)
        self.assertAlmostEqual(rouge_1.result().numpy(), 0.367, delta=1e-3)

        rouge_2.update_state(y_true_3, y_pred_3)
        self.assertAlmostEqual(rouge_2.result().numpy(), 0.222, delta=1e-3)

        merged_rouge = Rouge()
        merged_rouge.merge_state([rouge_1, rouge_2])
        self.assertAlmostEqual(merged_rouge.result().numpy(), 0.331, delta=1e-3)

    def test_get_config(self):
        rouge = Rouge(
            variant="rouge5",
            metric_type="precision",
            use_stemmer=True,
            dtype=tf.float32,
            name="rouge_test",
        )

        config = rouge.get_config()
        expected_config = {
            "variant": "rouge5",
            "metric_type": "precision",
            "use_stemmer": True,
            "dtype": tf.float32,
            "name": "rouge_test",
        }
        self.assertEqual(config, expected_config)
