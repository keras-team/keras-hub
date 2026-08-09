import keras
import pytest
import tensorflow as tf

from keras_hub.src.metrics.rouge_n import RougeN
from keras_hub.src.tests.test_case import TestCase


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
            "i really love contributing to KerasHub",
        ]
        y_pred = [
            "the cat was under the bed",
            "i love contributing to KerasHub",
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
                "i really love contributing to KerasHub",
            ]
        )
        y_pred = tf.constant(
            ["the cat was under the bed", "i love contributing to KerasHub"]
        )

        rouge_val = rouge(y_true, y_pred)
        self.assertAllClose(
            rouge_val,
            {"precision": 0.575, "recall": 0.4, "f1_score": 0.466666},
        )

    @pytest.mark.tf_only  # string model output only applies to tf.
    def test_model_compile(self):
        inputs = keras.Input(shape=(None,), dtype="string")
        outputs = keras.layers.Identity()(inputs)
        model = keras.Model(inputs, outputs)

        class EmptyLoss(keras.Loss):
            def __call__(self, y_true, y_pred, sample_weight):
                return 0.5

        model.compile(loss=EmptyLoss(), metrics=[RougeN()], run_eagerly=True)

        x = tf.constant([["hello this is fun"]])
        y = tf.constant([["hello this is awesome"]])

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
        y_true = [
            "the tiny little cat was found under the big funny bed",
            "i really love contributing to KerasHub",
        ]
        y_pred = [
            "the cat was under the bed",
            "i love contributing to KerasHub",
        ]

        rouge_val = rouge(y_true, y_pred)
        self.assertAllClose(
            rouge_val,
            {"precision": 0.333333, "recall": 0.25, "f1_score": 0.285714},
        )

    def test_reset_state(self):
        rouge = RougeN()
        y_true = ["hey, this is great fun", "i love contributing to KerasHub"]
        y_pred = [
            "great fun indeed",
            "KerasHub is awesome, i love contributing to it",
        ]

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
        y_true_1 = [
            "the tiny little cat was found under the big funny bed",
            "i really love contributing to KerasHub",
        ]
        y_pred_1 = [
            "the cat was under the bed",
            "i love contributing to KerasHub",
        ]

        rouge.update_state(y_true_1, y_pred_1)
        rouge_val = rouge.result()
        self.assertAllClose(
            rouge_val,
            {"precision": 0.575, "recall": 0.4, "f1_score": 0.466666},
        )

        y_true_2 = ["what is your favourite show"]
        y_pred_2 = ["my favourite show is silicon valley"]

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
