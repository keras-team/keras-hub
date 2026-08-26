import numpy as np
import tensorflow as tf

from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.tokenizers.unicode_codepoint_tokenizer import (
    UnicodeCodepointTokenizer,
)


class UnicodeCodepointTokenizerTest(TestCase):
    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=UnicodeCodepointTokenizer,
            init_kwargs={},
            input_data=["ninja", "samurai", "▀▁▂▃", "keras", "tensorflow"],
            expected_output=[
                [110, 105, 110, 106, 97],
                [115, 97, 109, 117, 114, 97, 105],
                [9600, 9601, 9602, 9603],
                [107, 101, 114, 97, 115],
                [116, 101, 110, 115, 111, 114, 102, 108, 111, 119],
            ],
        )

    def test_tokenizer_basics_with_sequence_length(self):
        # None of the inputs below get truncated at `sequence_length=10`, so
        # the dense, padded output round-trips cleanly through `detokenize`.
        self.run_preprocessing_layer_test(
            cls=UnicodeCodepointTokenizer,
            init_kwargs={"sequence_length": 10},
            input_data=["ninja", "samurai", "▀▁▂▃", "keras", "tensorflow"],
            expected_output=[
                [110, 105, 110, 106, 97, 0, 0, 0, 0, 0],
                [115, 97, 109, 117, 114, 97, 105, 0, 0, 0],
                [9600, 9601, 9602, 9603, 0, 0, 0, 0, 0, 0],
                [107, 101, 114, 97, 115, 0, 0, 0, 0, 0],
                [116, 101, 110, 115, 111, 114, 102, 108, 111, 119],
            ],
        )

    def test_tokenizer_basics_with_non_default_config(self):
        self.run_preprocessing_layer_test(
            cls=UnicodeCodepointTokenizer,
            init_kwargs={
                "lowercase": False,
                "sequence_length": 8,
                "normalization_form": "NFC",
                "errors": "ignore",
                "replacement_char": 0,
                "vocabulary_size": 100,
            },
            input_data=["ninja", "samurai", "▀▁▂▃"],
            expected_output=[
                [99, 99, 99, 99, 97, 0, 0, 0],
                [99, 97, 99, 99, 99, 97, 99, 0],
                [99, 99, 99, 99, 0, 0, 0, 0],
            ],
            expected_detokenize_output=[b"cccca", b"cacccac", b"cccc"],
        )

    def test_tokenize_scalar(self):
        input_data = "ninja"
        tokenizer = UnicodeCodepointTokenizer()
        call_output = tokenizer(input_data)
        tokenize_output = tokenizer.tokenize(input_data)

        self.assertAllEqual(call_output, [110, 105, 110, 106, 97])
        self.assertAllEqual(tokenize_output, [110, 105, 110, 106, 97])

    def test_dense_output(self):
        input_data = ["ninja", "samurai", "▀▁▂▃"]
        tokenizer = UnicodeCodepointTokenizer(sequence_length=10)
        call_output = tokenizer(input_data)
        self.assertAllEqual(
            call_output,
            [
                [110, 105, 110, 106, 97, 0, 0, 0, 0, 0],
                [115, 97, 109, 117, 114, 97, 105, 0, 0, 0],
                [9600, 9601, 9602, 9603, 0, 0, 0, 0, 0, 0],
            ],
        )

    def test_tokenize_scalar_with_vocabulary_size(self):
        input_data = "ninja"
        tokenizer = UnicodeCodepointTokenizer(vocabulary_size=105)
        call_output = tokenizer(input_data)
        tokenize_output = tokenizer.tokenize(input_data)

        self.assertAllEqual(call_output, [104, 104, 104, 104, 97])
        self.assertAllEqual(tokenize_output, [104, 104, 104, 104, 97])

    def test_tokenize_dense_with_vocabulary_size(self):
        input_data = ["ninja", "samurai", "▀▁▂▃"]
        tokenizer = UnicodeCodepointTokenizer(
            sequence_length=10, vocabulary_size=105
        )
        call_output = tokenizer(input_data)
        self.assertAllEqual(
            call_output,
            [
                [104, 104, 104, 104, 97, 0, 0, 0, 0, 0],
                [104, 97, 104, 104, 104, 97, 104, 0, 0, 0],
                [104, 104, 104, 104, 0, 0, 0, 0, 0, 0],
            ],
        )

    def test_tokenize_ragged_with_vocabulary_size(self):
        input_data = ["ninja", "samurai", "▀▁▂▃"]
        tokenizer = UnicodeCodepointTokenizer(vocabulary_size=105)
        call_output = tokenizer(input_data)
        tokenize_output = tokenizer.tokenize(input_data)
        exp_outputs = [
            [104, 104, 104, 104, 97],
            [104, 97, 104, 104, 104, 97, 104],
            [104, 104, 104, 104],
        ]
        self.assertAllEqual(call_output, exp_outputs)
        self.assertAllEqual(tokenize_output, exp_outputs)

    def test_detokenize(self):
        input_data = [
            [110, 105, 110, 106, 97],
            [115, 97, 109, 117, 114, 97, 105],
            [9600, 9601, 9602, 9603],
        ]

        tokenizer = UnicodeCodepointTokenizer()
        detokenize_output = tokenizer.detokenize(input_data)
        self.assertAllEqual(
            detokenize_output,
            [
                b"ninja",
                b"samurai",
                b"\xe2\x96\x80\xe2\x96\x81\xe2\x96\x82\xe2\x96\x83",
            ],
        )

    def test_detokenize_replace_error(self):
        # 10000000 is an invalid value
        input_data = tf.ragged.constant([[110, 105, 10000000, 110, 106, 97]])
        tokenizer = UnicodeCodepointTokenizer(
            errors="replace", replacement_char=75
        )
        detokenize_output = tokenizer.detokenize(input_data)
        self.assertAllEqual(detokenize_output, [b"niKnja"])

    def test_detokenize_ignore_error(self):
        input_data = tf.ragged.constant([[110, 105, 10000000, 110, 106, 97]])
        tokenizer = UnicodeCodepointTokenizer(errors="ignore")
        detokenize_output = tokenizer.detokenize(input_data)
        self.assertAllEqual(detokenize_output, [b"ninja"])

    def test_detokenize_strict_error(self):
        input_data = tf.ragged.constant([[110, 105, 10000000, 110, 106, 97]])
        tokenizer = UnicodeCodepointTokenizer(errors="strict")
        with self.assertRaises((ValueError, tf.errors.InvalidArgumentError)):
            _ = tokenizer.detokenize(input_data)

    def test_normalization_without_UTF8_valueerror(self):
        with self.assertRaises(ValueError):
            _ = UnicodeCodepointTokenizer(
                errors="strict",
                input_encoding="UTF-16",
                normalization_form="NFC",
            )

    def test_lowercase(self):
        input_data = tf.constant(["NiNJaS"])
        tokenizer = UnicodeCodepointTokenizer()
        call_output = tokenizer(input_data)
        self.assertAllEqual(
            call_output,
            [[110, 105, 110, 106, 97, 115]],
        )

    def test_skip_lowercase(self):
        input_data = tf.constant(["NiNJaS"])
        tokenizer = UnicodeCodepointTokenizer(lowercase=False)
        call_output = tokenizer(input_data)
        self.assertAllEqual(
            call_output,
            [[78, 105, 78, 74, 97, 83]],
        )

    def test_token_to_id(self):
        input_tokens = ["ب", "و", "خ"]
        expected_ids = [1576, 1608, 1582]
        tokenizer = UnicodeCodepointTokenizer(vocabulary_size=2000)
        ids = [tokenizer.token_to_id(t) for t in input_tokens]
        self.assertAllEqual(ids, expected_ids)

    def test_id_to_token(self):
        input_ids = [1576, 1608, 1582]
        expected_tokens = ["ب", "و", "خ"]
        tokenizer = UnicodeCodepointTokenizer(vocabulary_size=2000)
        tokens = [tokenizer.id_to_token(i) for i in input_ids]
        self.assertAllEqual(tokens, expected_tokens)

    def test_tokenize_tf(self):
        input_data = [
            "Hello World!",  # ASCII
            "¡Hola, señor!",  # Latin-1
            "Привет",  # Cyrillic
            "你好，世界",  # CJK
            "🚀🌟",  # Emoji (surrogate pairs in narrow builds, SMP in wide)
            b"Invalid \xff bytes".decode(
                "utf-8", "replace"
            ),  # Error replacement chars
            "Straße",  # German sharp S for casefolding
        ]

        tokenizer = UnicodeCodepointTokenizer(
            sequence_length=15, vocabulary_size=100000, errors="replace"
        )

        # Test Tokenization Parity
        tf_tokens = np.array(tokenizer._tokenize_tf(tf.constant(input_data)))
        py_tokens = tokenizer._tokenize_python(input_data)
        self.assertAllEqual(tf_tokens, py_tokens)

        # Test Detokenization Parity
        tf_detokenized = tokenizer._detokenize_tf(tf.constant(tf_tokens))
        py_detokenized = tokenizer._detokenize_python(py_tokens)
        self.assertAllEqual(tf_detokenized, py_detokenized)

        # Test empty batch parity and shape
        empty_tf = np.array(
            tokenizer._tokenize_tf(tf.constant([], dtype=tf.string))
        )
        empty_py = tokenizer._tokenize_python([])
        self.assertAllEqual(empty_tf, empty_py)
        self.assertEqual(empty_py.shape, (0, 15))
