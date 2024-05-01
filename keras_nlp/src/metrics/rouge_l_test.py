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

try:
    import tensorflow as tf
except ImportError:
    raise ImportError(
        "To use `keras_nlp`, please install Tensorflow: `pip install tensorflow`. "
        "The TensorFlow package is required for data preprocessing with any backend."
    )

from keras_nlp.src.metrics.rouge_l import RougeL
from keras_nlp.src.tests.test_case import TestCase


class RougeLTest(TestCase):
    def test_initialization(self):
        rouge = RougeL()
        result = rouge.result()

        self.assertDictEqual(
            result,
            {"precision": 0.0, "recall": 0.0, "f1_score": 0.0},
        )

    def test_string_input(self):
        rouge = RougeL(use_stemmer=False)
        y_true = "the tiny little cat was found under the big funny bed"
        y_pred = "the cat was under the bed"

        rouge_val = rouge(y_true, y_pred)
        self.assertAllClose(
            rouge_val,
            {"precision": 1.0, "recall": 0.545454, "f1_score": 0.705882},
        )

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
        self.assertAllClose(
            rouge_val,
            {"precision": 1.0, "recall": 0.689393, "f1_score": 0.807486},
        )

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
        self.assertAllClose(
            rouge_val,
            {"precision": 1.0, "recall": 0.689393, "f1_score": 0.807486},
        )

    def test_reset_state(self):
        rouge = RougeL()
        y_true = ["hey, this is great fun", "i love contributing to KerasNLP"]
        y_pred = [
            "great fun indeed",
            "KerasNLP is awesome, i love contributing to it",
        ]

        rouge.update_state(y_true, y_pred)
        rouge_val = rouge.result()
        self.assertNotAllClose(
            rouge_val,
            {"precision": 0.0, "recall": 0.0, "f1_score": 0.0},
        )

        rouge.reset_state()
        rouge_val = rouge.result()
        self.assertDictEqual(
            rouge_val,
            {"precision": 0.0, "recall": 0.0, "f1_score": 0.0},
        )

    def test_update_state(self):
        rouge = RougeL()
        y_true_1 = [
            "the tiny little cat was found under the big funny bed",
            "i really love contributing to KerasNLP",
        ]
        y_pred_1 = [
            "the cat was under the bed",
            "i love contributing to KerasNLP",
        ]

        rouge.update_state(y_true_1, y_pred_1)
        rouge_val = rouge.result()
        self.assertAllClose(
            rouge_val,
            {"precision": 1.0, "recall": 0.689393, "f1_score": 0.807486},
        )

        y_true_2 = ["what is your favourite show"]
        y_pred_2 = ["my favourite show is silicon valley"]

        rouge.update_state(y_true_2, y_pred_2)
        rouge_val = rouge.result()
        self.assertAllClose(
            rouge_val,
            {"precision": 0.777777, "recall": 0.592929, "f1_score": 0.659536},
        )

    def test_get_config(self):
        rouge = RougeL(
            use_stemmer=True,
            dtype="float32",
            name="rouge_l_test",
        )

        config = rouge.get_config()
        expected_config_subset = {
            "use_stemmer": True,
        }
        self.assertEqual(config, {**config, **expected_config_subset})
