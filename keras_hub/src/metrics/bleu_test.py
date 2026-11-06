import keras
import numpy as np
import pytest
import tensorflow as tf

from keras_hub.src.metrics.bleu import Bleu
from keras_hub.src.metrics.bleu import _tokenize_13a
from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.tokenizers.byte_tokenizer import ByteTokenizer


class BleuTest(TestCase):
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
        self.assertAlmostEqual(bleu_val, 0.212, delta=1e-3)

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
        self.assertAlmostEqual(bleu_val, 0.243, delta=1e-3)

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
        self.assertAlmostEqual(bleu_val, 0.243, delta=1e-3)

    def test_custom_tokenizer(self):
        byte_tokenizer = ByteTokenizer()
        bleu = Bleu(tokenizer=byte_tokenizer)
        y_true = [
            ["He eats a sweet apple."],
            ["Silicon Valley is one of my favourite shows!"],
        ]
        y_pred = [
            "He He He eats sweet apple which is a fruit.",
            "I love Silicon Valley, it's one of my favourite shows.",
        ]

        bleu_val = bleu(y_true, y_pred)
        self.assertAlmostEqual(bleu_val, 0.609, delta=1e-3)

    def test_default_tokenizer(self):
        # Expected outputs match SacreBLEU's `tokenizer_13a` (and the previous
        # `tf.strings.regex_replace` implementation of it).
        cases = {
            "He eats a sweet apple.": [
                "He",
                "eats",
                "a",
                "sweet",
                "apple",
                ".",
            ],
            "Prices rose 3.5% to $1,200.50 in 2024-25, then fell.": [
                "Prices",
                "rose",
                "3.5",
                "%",
                "to",
                "$",
                "1,200.50",
                "in",
                "2024",
                "-",
                "25",
                ",",
                "then",
                "fell",
                ".",
            ],
            "a-b 9-5 (x) [y] {z} <tag> &amp; &quot;q&quot;": [
                "a-b",
                "9",
                "-",
                "5",
                "(",
                "x",
                ")",
                "[",
                "y",
                "]",
                "{",
                "z",
                "}",
                "<",
                "tag",
                ">",
                "&",
                '"',
                "q",
                '"',
            ],
            "line1\nline2 hyphen-\nated": ["line1", "line2", "hyphenated"],
            "<skipped>text   many    spaces\t tabs": [
                "text",
                "many",
                "spaces",
                "tabs",
            ],
            "end with comma,": ["end", "with", "comma", ","],
            "Ünïcödé çhars: naïve café!": [
                "Ünïcödé",
                "çhars",
                ":",
                "naïve",
                "café",
                "!",
            ],
            "": [],
            "...": [".", ".", "."],
        }
        for text, expected in cases.items():
            self.assertEqual(_tokenize_13a(text), expected, msg=repr(text))

    def test_numpy_input(self):
        bleu = Bleu()
        y_true = np.array(
            [
                ["He eats a sweet apple."],
                ["Silicon Valley is one of my favourite shows!"],
            ]
        )
        y_pred = np.array(
            [
                "He He He eats sweet apple which is a fruit.",
                "I love Silicon Valley, it's one of my favourite shows.",
            ]
        )

        bleu_val = bleu(y_true, y_pred)
        self.assertAlmostEqual(bleu_val, 0.243, delta=1e-3)

    def test_ragged_references(self):
        # Different samples can have a different number of references, as a
        # nested Python list or as a `tf.RaggedTensor`.
        y_true = [
            [
                "He eats a sweet apple.",
                "He eats a sweet apple which is a fruit.",
            ],
            ["Silicon Valley is one of my favourite shows!"],
        ]
        y_pred = [
            "He He He eats sweet apple which is a fruit.",
            "I love Silicon Valley, it's one of my favourite shows.",
        ]
        bleu_val = Bleu()(y_true, y_pred)
        self.assertAlmostEqual(bleu_val, 0.493, delta=1e-3)
        bleu_val = Bleu()(tf.ragged.constant(y_true), tf.constant(y_pred))
        self.assertAlmostEqual(bleu_val, 0.493, delta=1e-3)

    def test_python_callable_tokenizer(self):
        bleu = Bleu(tokenizer=lambda text: text.lower().split())
        y_true = [["He eats a sweet apple."]]
        y_pred = ["he eats a sweet apple."]
        self.assertAlmostEqual(bleu(y_true, y_pred), 1.0, delta=1e-6)

    def test_different_order(self):
        bleu = Bleu(max_order=5)
        y_true = [
            ["He eats a sweet apple."],
            ["Silicon Valley is one of my favourite shows!"],
        ]
        y_pred = [
            "He He He eats sweet apple which is a fruit.",
            "I love Silicon Valley, it's one of my favourite shows.",
        ]

        bleu_val = bleu(y_true, y_pred)
        self.assertAlmostEqual(bleu_val, 0.188, delta=1e-3)

    def test_tensor_input(self):
        bleu = Bleu()
        y_true = tf.constant(
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
        self.assertAlmostEqual(bleu_val, 0.243, delta=1e-3)

    @pytest.mark.tf_only  # string model output only applies to tf.
    def test_model_compile(self):
        inputs = keras.Input(shape=(), dtype="string")
        outputs = keras.layers.Identity()(inputs)
        model = keras.Model(inputs, outputs)
        model.compile(metrics=[Bleu()])

        y_pred = x = tf.constant(
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

        output = model.compute_metrics(x, y, y_pred, sample_weight=None)
        self.assertAlmostEqual(output["bleu"], 0.243, delta=1e-3)

    def test_reset_state(self):
        bleu = Bleu()
        y_true = [
            ["He eats a sweet apple."],
            ["Silicon Valley is one of my favourite shows!"],
        ]
        y_pred = [
            "He He He eats sweet apple which is a fruit.",
            "I love Silicon Valley, it's one of my favourite shows.",
        ]

        bleu.update_state(y_true, y_pred)
        bleu_val = bleu.result()
        self.assertNotEqual(bleu_val, 0.0)

        bleu.reset_state()
        bleu_val = bleu.result()
        self.assertEqual(bleu_val, 0.0)

    def test_update_state(self):
        bleu = Bleu()
        y_true_1 = [
            ["He eats a sweet apple."],
            ["Silicon Valley is one of my favourite shows!"],
        ]
        y_pred_1 = [
            "He He He eats sweet apple which is a fruit.",
            "I love Silicon Valley, it's one of my favourite shows.",
        ]

        bleu.update_state(y_true_1, y_pred_1)
        bleu_val = bleu.result()
        self.assertAlmostEqual(bleu_val, 0.243, delta=1e-3)

        y_true_2 = ["Virat Kohli is the GOAT."]
        y_pred_2 = "Virat Kohli is the greatest of all time!"

        bleu.update_state(y_true_2, y_pred_2)
        bleu_val = bleu.result()
        self.assertAlmostEqual(bleu_val, 0.26, delta=1e-3)

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

    def test_invalid_input_rank(self):
        bleu = Bleu()
        # y_pred has base_rank 0. Rank 2 (base_rank + 2) is allowed ONLY if
        # last dim is 1.
        y_pred = tf.constant([["a", "b"], ["c", "d"]])
        y_true = [["a b"], ["c d"]]
        with self.assertRaisesRegex(ValueError, "y_pred is of rank 2"):
            bleu.update_state(y_true, y_pred)

        # y_true has base_rank 1. Rank 4 (base_rank + 3) is never allowed.
        y_true = tf.constant([[[["a"]]]])
        y_pred = ["a"]
        with self.assertRaisesRegex(
            ValueError, "y_true must be of rank 1, 2, or 3"
        ):
            bleu.update_state(y_true, y_pred)
