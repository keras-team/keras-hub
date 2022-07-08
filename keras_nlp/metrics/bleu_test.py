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

"""Tests for Bleu."""

import tensorflow as tf
from tensorflow import keras

from keras_nlp.metrics import Bleu
from keras_nlp.tokenizers import ByteTokenizer


class BleuTest(tf.test.TestCase):
    def test_initialization(self):
        bleu = Bleu()
        result = bleu.result()

        self.assertEqual(result, 0.0)

    def test_scalar_input(self):
        bleu = Bleu(smooth=True)
        y_true = [
            "He eats a sweet apple.",
            "He is eating a tasty apple, isn't he?",
        ]
        y_pred = "He He He eats sweet apple which is a fruit."

        bleu_val = bleu(y_true, y_pred)
        self.assertAlmostEqual(bleu_val.numpy(), 0.212, delta=1e-3)

    def test_1d_list_input(self):
        bleu = Bleu()
        y_true = [
            ["He eats a sweet apple."],
            ["Silicon Valley is one of my favourite shows!"],
        ]
        y_pred = [
            "He He He eats sweet apple which is a fruit.",
            "I love Silicon Valley, it's one of my favourite shows.",
        ]

        bleu_val = bleu(y_true, y_pred)
        self.assertAlmostEqual(bleu_val.numpy(), 0.243, delta=1e-3)

    def test_2d_list_input(self):
        bleu = Bleu()
        y_true = [
            [["He eats a sweet apple."]],
            [["Silicon Valley is one of my favourite shows!"]],
        ]
        y_pred = [
            ["He He He eats sweet apple which is a fruit."],
            ["I love Silicon Valley, it's one of my favourite shows."],
        ]

        bleu_val = bleu(y_true, y_pred)
        self.assertAlmostEqual(bleu_val.numpy(), 0.243, delta=1e-3)

    def test_1d_tensor_input(self):
        bleu = Bleu()
        y_true = tf.ragged.constant(
            [
                ["He eats a sweet apple."],
                ["Silicon Valley is one of my favourite shows!"],
            ]
        )
        y_pred = tf.constant(
            [
                "He He He eats sweet apple which is a fruit.",
                "I love Silicon Valley, it's one of my favourite shows.",
            ]
        )

        bleu_val = bleu(y_true, y_pred)
        self.assertAlmostEqual(bleu_val.numpy(), 0.243, delta=1e-3)

    def test_2d_tensor_input(self):
        bleu = Bleu()
        y_true = tf.constant(
            [
                [["He eats a sweet apple."]],
                [["Silicon Valley is one of my favourite shows!"]],
            ]
        )
        y_pred = tf.constant(
            [
                ["He He He eats sweet apple which is a fruit."],
                ["I love Silicon Valley, it's one of my favourite shows."],
            ]
        )

        bleu_val = bleu(y_true, y_pred)
        self.assertAlmostEqual(bleu_val.numpy(), 0.243, delta=1e-3)

    def test_custom_tokenizer(self):
        byte_tokenizer = ByteTokenizer()
        bleu = Bleu(tokenizer=byte_tokenizer)
        y_true = tf.ragged.constant(
            [
                ["He eats a sweet apple."],
                ["Silicon Valley is one of my favourite shows!"],
            ]
        )
        y_pred = tf.constant(
            [
                "He He He eats sweet apple which is a fruit.",
                "I love Silicon Valley, it's one of my favourite shows.",
            ]
        )

        bleu_val = bleu(y_true, y_pred)
        self.assertAlmostEqual(bleu_val.numpy(), 0.609, delta=1e-3)

    def test_different_order(self):
        bleu = Bleu(max_order=5)
        y_true = tf.ragged.constant(
            [
                ["He eats a sweet apple."],
                ["Silicon Valley is one of my favourite shows!"],
            ]
        )
        y_pred = tf.constant(
            [
                "He He He eats sweet apple which is a fruit.",
                "I love Silicon Valley, it's one of my favourite shows.",
            ]
        )

        bleu_val = bleu(y_true, y_pred)
        self.assertAlmostEqual(bleu_val.numpy(), 0.188, delta=1e-3)

    def test_model_compile(self):
        inputs = keras.Input(shape=(), dtype="string")
        outputs = tf.identity(inputs)
        model = keras.Model(inputs, outputs)

        model.compile(metrics=[Bleu()])

        x = tf.constant(
            [
                "He He He eats sweet apple which is a fruit.",
                "I love Silicon Valley, it's one of my favourite shows.",
            ]
        )
        y = tf.constant(
            [
                ["He eats a sweet apple."],
                ["Silicon Valley is one of my favourite shows!"],
            ]
        )

        output = model.evaluate(x, y, return_dict=True)
        self.assertAlmostEqual(output["bleu"], 0.243, delta=1e-3)

    def test_reset_state(self):
        bleu = Bleu()
        y_true = tf.ragged.constant(
            [
                ["He eats a sweet apple."],
                ["Silicon Valley is one of my favourite shows!"],
            ]
        )
        y_pred = tf.constant(
            [
                "He He He eats sweet apple which is a fruit.",
                "I love Silicon Valley, it's one of my favourite shows.",
            ]
        )

        bleu.update_state(y_true, y_pred)
        bleu_val = bleu.result()
        self.assertNotEqual(bleu_val.numpy(), 0.0)

        bleu.reset_state()
        bleu_val = bleu.result()
        self.assertEqual(bleu_val, 0.0)

    def test_update_state(self):
        bleu = Bleu()
        y_true_1 = tf.ragged.constant(
            [
                ["He eats a sweet apple."],
                ["Silicon Valley is one of my favourite shows!"],
            ]
        )
        y_pred_1 = tf.constant(
            [
                "He He He eats sweet apple which is a fruit.",
                "I love Silicon Valley, it's one of my favourite shows.",
            ]
        )

        bleu.update_state(y_true_1, y_pred_1)
        bleu_val = bleu.result()
        self.assertAlmostEqual(bleu_val.numpy(), 0.243, delta=1e-3)

        y_true_2 = tf.constant(["Virat Kohli is the GOAT."])
        y_pred_2 = tf.constant("Virat Kohli is the greatest of all time!")

        bleu.update_state(y_true_2, y_pred_2)
        bleu_val = bleu.result()
        self.assertAlmostEqual(bleu_val.numpy(), 0.26, delta=1e-3)

    def test_merge_state_normalize(self):
        bleu_1 = Bleu(smooth=True)
        bleu_2 = Bleu(smooth=True)

        y_true_1 = tf.ragged.constant(
            [
                ["He eats a sweet apple."],
                ["Silicon Valley is one of my favourite shows!"],
            ]
        )
        y_pred_1 = tf.constant(
            [
                "He He He eats sweet apple which is a fruit.",
                "I love Silicon Valley, it's one of my favourite shows.",
            ]
        )

        y_true_2 = tf.constant(["Virat Kohli is the GOAT."])
        y_pred_2 = tf.constant("Virat Kohli is the greatest of all time!")

        y_true_3 = tf.constant([["Watching Test cricket is so much fun."]])
        y_pred_3 = tf.constant(["Test is the best format in cricket."])

        bleu_1.update_state(y_true_1, y_pred_1)
        bleu_1.update_state(y_true_2, y_pred_2)
        bleu_val = bleu_1.result()
        self.assertAlmostEqual(bleu_val.numpy(), 0.293, delta=1e-3)

        bleu_2.update_state(y_true_3, y_pred_3)
        bleu_val = bleu_2.result()
        self.assertAlmostEqual(bleu_val.numpy(), 0.202, delta=1e-3)

        merged_bleu = Bleu(smooth=True)
        merged_bleu.merge_state([bleu_1, bleu_2])
        bleu_val = merged_bleu.result()
        self.assertAlmostEqual(bleu_val.numpy(), 0.495, delta=1e-3)

    def test_get_config(self):
        byte_tokenizer = ByteTokenizer()
        bleu = Bleu(
            tokenizer=byte_tokenizer,
            max_order=8,
            smooth=True,
            dtype=tf.float64,
            name="bleu_test",
        )

        config = bleu.get_config()
        expected_config_subset = {
            "tokenizer": byte_tokenizer,
            "max_order": 8,
            "smooth": True,
        }
        self.assertEqual(config, {**config, **expected_config_subset})
