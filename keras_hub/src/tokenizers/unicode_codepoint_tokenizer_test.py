import numpy as np
import pytest
import tensorflow as tf

from keras_hub.src.tests.test_case import TestCase
from keras_hub.src.tokenizers.unicode_codepoint_tokenizer import (
    UnicodeCodepointTokenizer,
)

try:
    import grain
except ImportError:
    grain = None
try:
    import tensorflow_text as tf_text
except ImportError:
    tf_text = None


class UnicodeCodepointTokenizerTest(TestCase):
    def setUp(self):
        super().setUp()
        self._allow_python_workflow = True

    def make_tokenizer(self, **kwargs):
        return UnicodeCodepointTokenizer(
            _allow_python_workflow=self._allow_python_workflow, **kwargs
        )

    def test_tokenizer_basics(self):
        self.run_preprocessing_layer_test(
            cls=UnicodeCodepointTokenizer,
            init_kwargs={
                "_allow_python_workflow": self._allow_python_workflow,
            },
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
            init_kwargs={
                "sequence_length": 10,
                "_allow_python_workflow": self._allow_python_workflow,
            },
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
                "_allow_python_workflow": self._allow_python_workflow,
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
        tokenizer = self.make_tokenizer()
        call_output = tokenizer(input_data)
        tokenize_output = tokenizer.tokenize(input_data)

        self.assertAllEqual(call_output, [110, 105, 110, 106, 97])
        self.assertAllEqual(tokenize_output, [110, 105, 110, 106, 97])

    def test_dense_output(self):
        input_data = ["ninja", "samurai", "▀▁▂▃"]
        tokenizer = self.make_tokenizer(sequence_length=10)
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
        tokenizer = self.make_tokenizer(vocabulary_size=105)
        call_output = tokenizer(input_data)
        tokenize_output = tokenizer.tokenize(input_data)

        self.assertAllEqual(call_output, [104, 104, 104, 104, 97])
        self.assertAllEqual(tokenize_output, [104, 104, 104, 104, 97])

    def test_tokenize_dense_with_vocabulary_size(self):
        input_data = ["ninja", "samurai", "▀▁▂▃"]
        tokenizer = self.make_tokenizer(sequence_length=10, vocabulary_size=105)
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
        tokenizer = self.make_tokenizer(vocabulary_size=105)
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

        tokenizer = self.make_tokenizer()
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
        tokenizer = self.make_tokenizer(errors="replace", replacement_char=75)
        detokenize_output = tokenizer.detokenize(input_data)
        self.assertAllEqual(detokenize_output, [b"niKnja"])

    def test_detokenize_ignore_error(self):
        input_data = tf.ragged.constant([[110, 105, 10000000, 110, 106, 97]])
        tokenizer = self.make_tokenizer(errors="ignore")
        detokenize_output = tokenizer.detokenize(input_data)
        self.assertAllEqual(detokenize_output, [b"ninja"])

    def test_detokenize_strict_error(self):
        input_data = tf.ragged.constant([[110, 105, 10000000, 110, 106, 97]])
        tokenizer = self.make_tokenizer(errors="strict")
        if self._allow_python_workflow:
            error = ValueError
        else:
            error = tf.errors.InvalidArgumentError
        with self.assertRaises(error):
            _ = tokenizer.detokenize(input_data)

    def test_normalization_without_UTF8_valueerror(self):
        with self.assertRaises(ValueError):
            _ = self.make_tokenizer(
                errors="strict",
                input_encoding="UTF-16",
                normalization_form="NFC",
            )

    def test_lowercase(self):
        input_data = tf.constant(["NiNJaS"])
        tokenizer = self.make_tokenizer()
        call_output = tokenizer(input_data)
        self.assertAllEqual(
            call_output,
            [[110, 105, 110, 106, 97, 115]],
        )

    def test_skip_lowercase(self):
        input_data = tf.constant(["NiNJaS"])
        tokenizer = self.make_tokenizer(lowercase=False)
        call_output = tokenizer(input_data)
        self.assertAllEqual(
            call_output,
            [[78, 105, 78, 74, 97, 83]],
        )

    def test_token_to_id(self):
        input_tokens = ["ب", "و", "خ"]
        expected_ids = [1576, 1608, 1582]
        tokenizer = self.make_tokenizer(vocabulary_size=2000)
        ids = [tokenizer.token_to_id(t) for t in input_tokens]
        self.assertAllEqual(ids, expected_ids)

    def test_id_to_token(self):
        input_ids = [1576, 1608, 1582]
        expected_tokens = ["ب", "و", "خ"]
        tokenizer = self.make_tokenizer(vocabulary_size=2000)
        tokens = [tokenizer.id_to_token(i) for i in input_ids]
        self.assertAllEqual(tokens, expected_tokens)

    def test_bytes_and_tensor_inputs(self):
        if not self._allow_python_workflow:
            self.skipTest("The TF path does not accept numpy string arrays.")
        tokenizer = self.make_tokenizer()
        expected = [[110, 105, 110, 106, 97], [9600, 9601]]
        self.assertAllEqual(tokenizer([b"ninja", "▀▁".encode()]), expected)
        self.assertAllEqual(tokenizer(np.array(["ninja", "▀▁"])), expected)
        self.assertAllEqual(tokenizer(tf.constant(["ninja", "▀▁"])), expected)
        self.assertAllEqual(tokenizer(b"ninja"), expected[0])
        self.assertAllEqual(tokenizer(tf.constant("ninja")), expected[0])

    def test_tokenize_invalid_bytes(self):
        input_data = [b"ni\xe2\x96ja", b"\xff", b"ok"]
        tokenizer = self.make_tokenizer(errors="replace", replacement_char=75)
        self.assertAllEqual(
            tokenizer(input_data), [[110, 105, 75, 106, 97], [75], [111, 107]]
        )
        tokenizer = self.make_tokenizer(errors="ignore")
        self.assertAllEqual(
            tokenizer(input_data), [[110, 105, 106, 97], [], [111, 107]]
        )

    def test_detokenize_inputs(self):
        if not self._allow_python_workflow:
            self.skipTest("The TF path does not accept int64 inputs.")
        tokenizer = self.make_tokenizer()
        ids = [[110, 105, 110, 106, 97, 0, 0], [9600, 9601, 0, 0, 0, 0, 0]]
        self.assertAllEqual(tokenizer.detokenize(ids), ["ninja", "▀▁"])
        self.assertAllEqual(
            tokenizer.detokenize(np.array(ids)), ["ninja", "▀▁"]
        )
        self.assertAllEqual(
            tokenizer.detokenize(tf.constant(ids)), ["ninja", "▀▁"]
        )
        self.assertAllEqual(tokenizer.detokenize(ids[0]), "ninja")
        self.assertAllEqual(tokenizer.detokenize(np.array(ids[0])), "ninja")

    @pytest.mark.skipif(grain is None, reason="grain is not installed")
    def test_grain_outputs_numpy(self):
        input_data = ["ninja", "▀▁"]
        # Ragged outputs are python lists.
        tokenizer = self.make_tokenizer()
        (outputs,) = list(grain.MapDataset.source([input_data]).map(tokenizer))
        self.assertIsInstance(outputs, list)
        self.assertEqual(outputs, [[110, 105, 110, 106, 97], [9600, 9601]])
        # Dense outputs are numpy arrays.
        tokenizer = self.make_tokenizer(sequence_length=6)
        outputs = list(grain.MapDataset.source(input_data).map(tokenizer))
        for output in outputs:
            self.assertIsInstance(output, np.ndarray)
        (batch,) = list(
            grain.MapDataset.source(input_data).map(tokenizer).batch(2)
        )
        self.assertAllEqual(
            batch, [[110, 105, 110, 106, 97, 0], [9600, 9601, 0, 0, 0, 0]]
        )
        self.assertAllEqual(batch, tokenizer(input_data))

    @pytest.mark.skipif(tf_text is None, reason="tensorflow-text not installed")
    def test_python_path_matches_tf_path(self):
        if not self._allow_python_workflow:
            self.skipTest("Parity test only runs from the python path.")
        corpus = [
            "The quick brown fox.",
            "ﬁsh ǅ ß İ Σ",
            "á é í ó ú Ç",
            "  leading and trailing  ",
            "x\xa0y\u3000z\ty\nq",
            "emoji 😀 test 한국어 テスト",
            "",
        ]
        bad_ids = [[110, 105, 10000000, 110], [110, -1, 110], [110, 0, 110]]
        bad_bytes = [b"ni\xe2\x96ja", b"\xff", b"HeLLo"]
        for kwargs in [
            {},
            {"lowercase": False},
            {"normalization_form": "NFKD"},
            {"lowercase": False, "normalization_form": "NFC"},
            {"sequence_length": 8},
            {"vocabulary_size": 105},
            {"errors": "replace", "replacement_char": 75},
            {"errors": "ignore"},
        ]:
            python_tokenizer = UnicodeCodepointTokenizer(**kwargs)
            tf_tokenizer = UnicodeCodepointTokenizer(
                _allow_python_workflow=False, **kwargs
            )
            python_ids = python_tokenizer(corpus)
            tf_ids = tf_tokenizer(corpus)
            self.assertAllEqual(python_ids, tf_ids)
            self.assertAllEqual(
                python_tokenizer.detokenize(python_ids),
                tf_tokenizer.detokenize(tf_ids),
            )
            self.assertAllEqual(
                python_tokenizer(bad_bytes), tf_tokenizer(bad_bytes)
            )
            if "sequence_length" not in kwargs:
                self.assertAllEqual(
                    python_tokenizer.detokenize(bad_ids),
                    tf_tokenizer.detokenize(tf.ragged.constant(bad_ids)),
                )


class UnicodeCodepointTokenizerTFTest(UnicodeCodepointTokenizerTest):
    """Set `_allow_python_workflow=False` to test TF execution."""

    def setUp(self):
        super().setUp()
        self._allow_python_workflow = False
